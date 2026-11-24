using System.Globalization;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class AudixaRegistration
{
    public static IServiceCollection AddAudixa(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AudixaProvider.Id, "https://api.audixa.ai/");
        ProviderServiceCollectionExtensions.AddClient(services, AudixaProvider.AssetClient, "https://api.audixa.ai/");
        return services.AddProvider<AudixaProvider>(AudixaProvider.Id);
    }
}

internal sealed partial class AudixaProvider(IHttpClientFactory clients, IProviderCredentials credentials, IConfiguration configuration)
    : IModelListingProvider, ISpeechProvider, IStreamingSpeechEventsProvider, IChatAudioRoutingProvider
{
    internal const string Id = "audixa";
    internal const string AssetClient = "audixa-assets";
    private static readonly string[] Models = ["base", "advanced"];

    public bool IsSpeechModel(string model) => TryModel(model, out _, out _);
    public bool IsTranscriptionModel(string model) => false;

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        foreach (var model in Models)
            yield return new Model { Id = model, Name = $"Audixa {Title(model)}", OwnedBy = "Audixa", Type = "speech" };
        if (credentials.GetApiKey(Id) is not { } key || string.IsNullOrWhiteSpace(key)) yield break;

        using var client = clients.CreateClient(Id);
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        const int limit = 100;
        for (var offset = 0; ;)
        {
            var response = await NativeHttp.GetAsync<JsonElement>(client, $"v3/voices?limit={limit}&offset={offset}", Auth(key), ct);
            var data = response.ValueKind == JsonValueKind.Array ? response : VoiceArray(response);
            if (data.ValueKind != JsonValueKind.Array || data.GetArrayLength() == 0) yield break;
            foreach (var voice in data.EnumerateArray())
            {
                var model = ReadString(voice, "model")?.Trim().ToLowerInvariant();
                var voiceId = (ReadString(voice, "voice_id") ?? ReadString(voice, "voiceId") ?? ReadString(voice, "id"))?.Trim();
                if (model is not ("base" or "advanced") || string.IsNullOrEmpty(voiceId) || voiceId.Contains('/')) continue;
                var id = $"{model}/{voiceId}";
                if (!seen.Add(id)) continue;
                var gender = ReadString(voice, "gender");
                var accent = ReadString(voice, "accent");
                var title = Title(model);
                var description = ReadString(voice, "description");
                yield return new Model { Id = id, OwnedBy = "Audixa", Type = "speech",
                    Name = $"{title} {ReadString(voice, "name") ?? voiceId} ({voiceId})",
                    Description = string.IsNullOrWhiteSpace(description)
                        ? $"Audixa {title} voice {voiceId} ({accent ?? "unknown accent"}, {gender ?? "unknown gender"})."
                        : $"Audixa {title} voice {voiceId}: {description}", Tags = string.IsNullOrWhiteSpace(gender) ? ["voice"] : ["voice", gender] };
            }
            var count = int.TryParse(ReadString(response, "length"), NumberStyles.None, CultureInfo.InvariantCulture, out var length)
                ? length : data.GetArrayLength();
            if (count < limit || offset > int.MaxValue - count) yield break;
            offset += count;
        }
    }

    private static JsonElement VoiceArray(JsonElement value)
    {
        foreach (var name in new[] { "voices", "data", "items", "results" })
            if (Find(value, name) is { ValueKind: JsonValueKind.Array } data) return data;
        return default;
    }

    private static string Title(string model) => char.ToUpperInvariant(model[0]) + model[1..];

    private static bool TryModel(string? value, out string model, out string? voice)
    {
        var parts = (value ?? "").Split('/', StringSplitOptions.TrimEntries);
        model = parts[0].ToLowerInvariant();
        voice = parts.Length == 2 ? parts[1] : null;
        return model is "base" or "advanced" && (parts.Length == 1 || parts.Length == 2 && voice!.Length > 0);
    }

    private static JsonElement Find(JsonElement value, string name)
    {
        if (value.ValueKind == JsonValueKind.Object)
            foreach (var property in value.EnumerateObject())
                if (property.Name.Equals(name, StringComparison.OrdinalIgnoreCase)) return property.Value;
        return default;
    }

    private static string? ReadString(JsonElement value, string name) => Find(value, name) switch
    {
        { ValueKind: JsonValueKind.String } property => property.GetString(),
        { ValueKind: JsonValueKind.Number } property => property.GetRawText(),
        _ => null
    };

    private static Action<HttpRequestHeaders> Auth(string key) => headers => headers.Add("x-api-key", key);

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var payload = Payload(request);
        var auth = Auth(RequiredKey(credentials, Id));
        using var client = clients.CreateClient(Id);
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromSeconds(60));
        var status = await NativeHttp.PostAsync<Dictionary<string, JsonElement>, JsonElement>(client, "v3/tts", payload, auth, headers, Json, deadline.Token);
        var id = ReadString(status, "generation_id");
        if (string.IsNullOrWhiteSpace(id)) throw new JsonException("Audixa did not return a generation ID.");
        var poll = $"v3/tts?generation_id={Uri.EscapeDataString(id)}";
        while (true)
        {
            var state = ReadString(status, "status")?.ToUpperInvariant();
            if (state == "COMPLETED" && ReadString(status, "audio_url") is { Length: > 0 } url)
                return await DownloadAsync(url, id, status, payload, deadline.Token);
            if (state == "FAILED") throw new HttpRequestException("Audixa TTS failed: " + (ReadString(status, "error_message") ?? ReadString(status, "detail") ?? "unknown error"));
            if (state is not (null or "IN_QUEUE" or "GENERATING")) throw new JsonException($"Audixa returned an invalid TTS status: {state}.");
            await Task.Delay(TimeSpan.FromSeconds(1), deadline.Token);
            status = await NativeHttp.GetAsync<JsonElement>(client, poll, auth, deadline.Token);
        }
    }

    private async Task<SpeechProviderResponse> DownloadAsync(string url, string id, JsonElement status,
        Dictionary<string, JsonElement> payload, CancellationToken ct)
    {
        using var client = clients.CreateClient(AssetClient);
        if (!Uri.TryCreate(client.BaseAddress, url, out var uri) || uri.Scheme != Uri.UriSchemeHttps || !uri.IsDefaultPort
            || uri.UserInfo.Length != 0 || uri.Fragment.Length != 0 || !IsAudioHost(uri.IdnHost))
            throw new JsonException("Audixa returned an untrusted audio URL.");
        // Downloads are isolated from both client and provider credentials; redirects are disabled.
        using var request = new HttpRequestMessage(HttpMethod.Get, uri);
        using var response = await NativeHttp.SendAsync(client, request, ct);
        var type = response.Content.Headers.ContentType?.MediaType;
        if (type is null or "application/octet-stream")
        {
            var format = payload.TryGetValue("audio_format", out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
            type = (format?.ToLowerInvariant() ?? Path.GetExtension(uri.AbsolutePath).Trim('.').ToLowerInvariant()) == "mp3" ? "audio/mpeg" : "audio/wav";
        }
        if (!type.StartsWith("audio/", StringComparison.OrdinalIgnoreCase)) throw new JsonException("Audixa did not return binary audio.");
        await response.Content.LoadIntoBufferAsync(32 * 1024 * 1024, ct);
        var bytes = await response.Content.ReadAsByteArrayAsync(ct);
        var content = new ByteArrayContent(bytes);
        content.Headers.ContentType = new(type);
        return new SpeechProviderResponse(content, Metadata(status, id));
    }

    private bool IsAudioHost(string host) => host == "audixa.ai" || host.EndsWith(".audixa.ai", StringComparison.OrdinalIgnoreCase)
        || configuration.GetSection("Providers:audixa:AudioHosts").GetChildren().Any(entry => string.Equals(host, entry.Value, StringComparison.OrdinalIgnoreCase));

    private static Dictionary<string, JsonElement> Metadata(JsonElement status, string? generationId = null)
    {
        // The audio URL may carry a signed URL token. Pass the other native status fields through unchanged.
        var raw = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (var property in status.EnumerateObject())
            if (!property.Name.Equals("audio_url", StringComparison.OrdinalIgnoreCase)) raw[property.Name] = property.Value;
        if (generationId is not null) raw.TryAdd("generation_id", JsonSerializer.SerializeToElement(generationId));
        return new() { [Id] = JsonSerializer.SerializeToElement(raw, Json) };
    }

    private static Dictionary<string, JsonElement> Payload(AudioSpeechRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Input)) throw new ProviderRequestException("Audixa requires text input.");
        if (!TryModel(request.Model, out var model, out var shortcut))
            throw new ProviderRequestException("Audixa model must be 'base', 'advanced', or one of those models followed by '/{voiceId}'.");
        var body = new Dictionary<string, JsonElement>(StringComparer.OrdinalIgnoreCase);
        if (request.AdditionalProperties is { } options)
        {
            foreach (var (name, value) in options)
                if (!name.Equals(Id, StringComparison.OrdinalIgnoreCase)) body[name] = value;
            if (options.TryGetValue(Id, out var native))
            {
                if (native.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Audixa options must be an object.");
                foreach (var property in native.EnumerateObject()) body[property.Name] = property.Value;
            }
        }
        var voice = (shortcut ?? request.Voice ?? TakeString(body, "voice_id") ?? TakeString(body, "voice"))?.Trim();
        if (string.IsNullOrWhiteSpace(voice)) throw new ProviderRequestException("Audixa requires a voice or a model/voice shortcut.");
        var language = TakeString(body, "language");
        foreach (var name in new[] { "model", "text", "input", "voice", "voice_id", "response_format", "stream_format", "stream", "language" }) body.Remove(name);
        body["text"] = JsonSerializer.SerializeToElement(request.Input);
        body["model"] = JsonSerializer.SerializeToElement(model);
        body["voice_id"] = JsonSerializer.SerializeToElement(voice);
        if (request.Speed is { } speed) body["speed"] = JsonSerializer.SerializeToElement(speed);
        if (!string.IsNullOrWhiteSpace(request.ResponseFormat)) body["audio_format"] = JsonSerializer.SerializeToElement(request.ResponseFormat.Trim());
        if (!string.IsNullOrWhiteSpace(language)) body["language_code"] = JsonSerializer.SerializeToElement(language.Trim());
        return body;
    }

    private static string? TakeString(Dictionary<string, JsonElement> options, string name)
        => options.TryGetValue(name, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
}
