using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class TypecastProviderRegistration
{
    public static IServiceCollection AddTypecast(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, TypecastProvider.Id, "https://api.typecast.ai/");
        return services.AddProvider<TypecastProvider>(TypecastProvider.Id);
    }
}

internal sealed class TypecastProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, ISpeechProvider, IChatAudioRoutingProvider
{
    internal const string Id = "typecast";
    private static readonly HashSet<string> Reserved = new(StringComparer.OrdinalIgnoreCase)
    {
        "authorization", "api_key", "apiKey", "x-api-key", "model", "voice_id", "text", "input", "voice", "response_format", "stream", "stream_format",
        "language", "prompt", "output", "emotionType", "emotion", "previousText", "nextText", "volume", "audioPitch", "audioTempo", "seed"
    };

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var key = credentials.GetApiKey(Id);
        if (string.IsNullOrWhiteSpace(key)) yield break;
        using var client = clients.CreateClient(Id);
        var voices = await NativeHttp.GetAsync<JsonElement>(client, "v2/voices", Authenticate(key), ct);
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var voice in Array(voices))
        {
            var voiceId = (String(voice, "voice_id") ?? String(voice, "voiceId") ?? String(voice, "id"))?.Trim();
            if (string.IsNullOrEmpty(voiceId) || voiceId.Contains('/')) continue;
            var name = String(voice, "voice_name") ?? String(voice, "voiceName") ?? voiceId;
            var gender = String(voice, "gender");
            var age = String(voice, "age")?.Replace('_', ' ');
            var useCases = Property(voice, "use_cases");
            if (useCases.ValueKind == JsonValueKind.Undefined) useCases = Property(voice, "useCases");
            var uses = useCases.ValueKind == JsonValueKind.String ? useCases.GetString() : string.Join(", ",
                Array(useCases).Where(item => item.ValueKind == JsonValueKind.String).Select(item => item.GetString()).Take(4));
            var details = $"{age ?? "unknown age"}, {gender ?? "unknown gender"}, use cases: {(string.IsNullOrWhiteSpace(uses) ? "general" : uses)}";
            foreach (var model in Array(Property(voice, "models")))
            {
                if (String(model, "version")?.Trim() is not { Length: > 0 } version || !seen.Add($"{version}/{voiceId}")) continue;
                yield return new Model
                {
                    Id = $"{version}/{voiceId}", OwnedBy = "Typecast", Type = "speech", Name = $"{version} / {name} ({details})",
                    Description = $"Typecast voice '{name}' (voice_id={voiceId}) on model '{version}' ({details}).",
                    Tags = string.IsNullOrWhiteSpace(gender) ? ["voice"] : ["voice", gender]
                };
            }
        }
    }

    public bool IsSpeechModel(string model) => SplitModel(model) is not null;
    public bool IsTranscriptionModel(string model) => false;

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var (model, voice) = SplitModel(request.Model) ?? throw new ProviderRequestException("Use a Typecast model ID with a model version and voice ID.");
        if (string.IsNullOrWhiteSpace(request.Input)) throw new ProviderRequestException("Typecast speech requires nonempty text.");
        if (request.StreamFormat == "sse") throw new ProviderRequestException("Typecast does not support native speech SSE.");

        var options = new Dictionary<string, JsonElement>(StringComparer.OrdinalIgnoreCase);
        var payload = new JsonObject();
        foreach (var (key, value) in request.AdditionalProperties ?? [])
        {
            if (Reserved.Contains(key)) options[key] = value;
            else payload[key] = Copy(value);
        }
        payload["model"] = model;
        payload["voice_id"] = voice;
        payload["text"] = request.Input;
        if (ReadString(options, "language")?.Trim() is { Length: > 0 } language) payload["language"] = language;
        if (options.TryGetValue("seed", out var seed)) payload["seed"] = Copy(seed);

        var prompt = ReadObject(options, "prompt");
        SetPrompt("emotionType", "emotion_type");
        SetPrompt("emotion", "emotion");
        SetPrompt("previousText", "previous_text");
        SetPrompt("nextText", "next_text");
        if (model.Equals("ssfm-v30", StringComparison.OrdinalIgnoreCase) && !prompt.ContainsKey("emotion_type"))
            prompt["emotion_type"] = "smart";
        if (prompt.Count > 0 || options.ContainsKey("prompt")) payload["prompt"] = prompt;

        var output = ReadObject(options, "output");
        SetNumber("volume", "volume", 100, 0, 200, integral: true);
        SetNumber("audioPitch", "audio_pitch", 0, -12, 12, integral: true);
        SetNumber("audioTempo", "audio_tempo", 1, 0.5, 2);
        var format = string.IsNullOrWhiteSpace(request.ResponseFormat)
            ? output["audio_format"] is JsonValue configuredFormat && configuredFormat.TryGetValue<string>(out var nativeFormat) ? nativeFormat : "wav"
            : request.ResponseFormat;
        format = format.Trim().ToLowerInvariant() is "mp3" or "mpeg" ? "mp3" : "wav";
        if (!output.ContainsKey("audio_format") || !string.IsNullOrWhiteSpace(request.ResponseFormat)) output["audio_format"] = format;
        payload["output"] = output;

        using var client = clients.CreateClient(Id);
        using var message = new HttpRequestMessage(HttpMethod.Post, "v1/text-to-speech") { Content = JsonContent.Create(payload, options: Json) };
        Authenticate(RequiredKey(credentials, Id))(message.Headers);
        NativeHttp.ApplyHeaders(message, headers);
        using var upstream = await NativeHttp.SendAsync(client, message, ct);
        var audio = upstream.Content;
        var mime = audio.Headers.ContentType?.MediaType;
        if (mime is null or "application/octet-stream") audio.Headers.ContentType = new(format == "mp3" ? "audio/mpeg" : "audio/wav");
        else if (!mime.StartsWith("audio/", StringComparison.OrdinalIgnoreCase)) throw new IOException("Typecast did not return audio.");
        var safeHeaders = NativeHttp.SafeResponseMetadataHeaders(upstream.Headers.ToDictionary(pair => pair.Key, pair => (string?)string.Join(", ", pair.Value),
            StringComparer.OrdinalIgnoreCase));
        upstream.Content = null;
        return new SpeechProviderResponse(audio, safeHeaders.Count == 0 ? null : new() { [Id] = JsonSerializer.SerializeToElement(new { headers = safeHeaders }, Json) });

        void SetPrompt(string alias, string name)
        {
            if (ReadString(options, alias)?.Trim() is { Length: > 0 } text) prompt[name] = text;
        }

        void SetNumber(string alias, string name, int fallback, double min, double max, bool integral = false)
        {
            if (!options.TryGetValue(alias, out var value)) { if (!output.ContainsKey(name)) output[name] = fallback; return; }
            if (value.ValueKind == JsonValueKind.Null) { output[name] = fallback; return; }
            if (value.ValueKind != JsonValueKind.Number || !value.TryGetDouble(out var number)
                || !double.IsFinite(number) || number < min || number > max || integral && number != Math.Truncate(number))
                throw new ProviderRequestException($"Typecast '{alias}' must be {(integral ? "an integer" : "a number")} between {min} and {max}.");
            output[name] = Copy(value);
        }
    }

    private static (string Model, string Voice)? SplitModel(string model)
    {
        var slash = model.LastIndexOf('/');
        return slash > 0 && model[..slash].Trim() is { Length: > 0 } version && model[(slash + 1)..].Trim() is { Length: > 0 } voice
            ? (version, voice) : null;
    }

    private static Action<HttpRequestHeaders> Authenticate(string key) => headers => headers.Add("X-API-KEY", key);
    private static JsonObject ReadObject(Dictionary<string, JsonElement> options, string name)
        => !options.TryGetValue(name, out var value) ? [] : Copy(value) as JsonObject
            ?? throw new ProviderRequestException($"Typecast '{name}' must be an object.");
    private static string? ReadString(Dictionary<string, JsonElement> options, string name)
        => !options.TryGetValue(name, out var value) || value.ValueKind == JsonValueKind.Null ? null
            : value.ValueKind == JsonValueKind.String ? value.GetString() : throw new ProviderRequestException($"Typecast '{name}' must be a string.");
}
