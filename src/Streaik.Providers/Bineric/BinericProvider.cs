using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Xml;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class BinericRegistration
{
    public static IServiceCollection AddBineric(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, "bineric", "https://api.bineric.com/api/v1/ai/");
        return services.AddProvider<BinericProvider>("bineric");
    }
}

internal sealed class BinericProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, "bineric"), INativeChatOptionsProvider, IChatAudioRoutingProvider,
      IStreamingSpeechProvider, ITranscriptionProvider
{
    private const string Speech = "text-to-speech";
    private const string Transcription = "speech-to-text";
    private const int MaxAudioBytes = 25 * 1024 * 1024;
    private static readonly HashSet<string> MediaControls = new(StringComparer.OrdinalIgnoreCase)
    { "bineric", "model", "messages", "input", "file", "stream", "stream_format", "response_format", "authorization", "api-key", "x-api-key", "api_key", "apiKey", "headers" };

    public bool IsSpeechModel(string model) => model.Equals(Speech, StringComparison.OrdinalIgnoreCase);
    public bool IsTranscriptionModel(string model) => model.Equals(Transcription, StringComparison.OrdinalIgnoreCase);

    protected override Action<HttpRequestHeaders> ChatAuth() => Authenticate(ApiKey());
    protected override async Task<JsonElement> FetchModelsAsync(string key, CancellationToken ct)
    {
        using var client = CreateClient();
        return await NativeHttp.GetAsync<JsonElement>(client, "models", Authenticate(key), ct);
    }

    private static Action<HttpRequestHeaders> Authenticate(string key) => headers => headers.Add("api-key", key);

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(ConfiguredApiKey())) yield break;
        var raw = await FetchModelsAsync(ApiKey(), ct);
        var hasSpeech = false;
        foreach (var item in Array(Property(raw, "data")))
        {
            if (String(item, "id") is not { Length: > 0 } id) continue;
            var name = String(item, "name") ?? id;
            var type = IsSpeechModel(id) || name == "Text to Speech" ? "speech"
                : IsTranscriptionModel(id) || name == "Speech to Text" ? "transcription" : "model";
            hasSpeech |= type == "speech";
            yield return new() { Id = id, Name = name, Type = type, OwnedBy = String(item, "provider") ?? ProviderId,
                Description = String(item, "description") };
        }
        if (!hasSpeech) yield return new() { Id = Speech, Name = "Text to Speech", Type = "speech", OwnedBy = ProviderId };
    }

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Input)) throw new ProviderRequestException("Bineric speech requires input text.");
        var body = MediaOptions(request.AdditionalProperties);
        var options = body["options"] is JsonObject native ? (JsonObject)native.DeepClone() : new JsonObject();
        foreach (var field in new[] { "language", "voice", "pitch", "rate", "volume" })
            foreach (var entry in body.Where(entry => entry.Key.Equals(field, StringComparison.OrdinalIgnoreCase)).ToArray())
            {
                options[field] = entry.Value?.DeepClone();
                body.Remove(entry.Key);
            }
        if (request.Voice is { } voice) options["voice"] = voice;
        if (request.Speed is { } rate) options["rate"] = rate;
        body["model"] = Speech;
        body["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = request.Input });
        body["options"] = options;

        var (raw, _) = await PostMediaAsync(body, headers, ct);
        var content = AssistantContent(raw);
        if (String(content, "audioContent") is not { } encoded) throw new JsonException("Bineric did not return speech audio.");
        encoded = encoded.Trim();
        var providerFormat = String(Property(content, "metadata"), "format");
        string? dataType = null;
        if (encoded.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
        {
            var comma = encoded.IndexOf(',');
            if (comma < 12 || !encoded[..comma].EndsWith(";base64", StringComparison.OrdinalIgnoreCase))
                throw new JsonException("Bineric returned invalid speech audio.");
            dataType = encoded[5..(comma - 7)];
            encoded = encoded[(comma + 1)..];
        }
        if (encoded.Length > MaxAudioBytes * 4 / 3 + 8) throw new JsonException("Bineric speech output is too large.");
        byte[] bytes;
        try { bytes = Convert.FromBase64String(encoded); }
        catch (FormatException ex) { throw new JsonException("Bineric returned invalid speech audio.", ex); }
        if (bytes.Length is 0 or > MaxAudioBytes) throw new JsonException("Bineric returned an invalid speech output size.");

        var mime = providerFormat?.Trim().ToLowerInvariant() switch
        {
            "wav" => "audio/wav", "opus" or "ogg" => "audio/ogg", "flac" => "audio/flac",
            "aac" => "audio/aac", "pcm" => "audio/pcm", "mp3" or "mpeg" => "audio/mpeg",
            _ => dataType is "audio/wav" or "audio/ogg" or "audio/flac" or "audio/aac" or "audio/pcm" ? dataType : "audio/mpeg"
        };
        var audio = new ByteArrayContent(bytes);
        audio.Headers.ContentType = new(mime);
        var metadata = JsonObject.Create(raw)!;
        // Keep native metadata but do not duplicate the audio in every gateway envelope.
        if (metadata["choices"] is JsonArray choices)
            foreach (var choice in choices.OfType<JsonObject>())
                if (choice["message"] is JsonObject message && message["content"] is JsonObject audioContent)
                    audioContent.Remove("audioContent");
        return new SpeechProviderResponse(audio, Metadata(JsonSerializer.SerializeToElement(metadata, Json)));
    }

    // Bineric returns one completed JSON result; the gateway frames its audio as SSE when requested.
    public Task<HttpResponseMessage> SpeakStreamingAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeakAsync(request, headers, ct);

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var encoded = await BoundedBase64Async(audio, MaxAudioBytes, ct);
        var body = MediaOptions(options);
        if (!string.IsNullOrWhiteSpace(language)) body["language"] = language;
        body["model"] = Transcription;
        body["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = "Transcribe this audio",
            ["attachments"] = new JsonArray(JsonValue.Create($"data:{mediaType};base64,{encoded}")) });
        var (raw, upstreamHeaders) = await PostMediaAsync(body, headers, ct);
        var native = Property(raw, "_bineric");
        var segments = new List<TranscriptionSegment>();
        foreach (var phrase in Array(Property(native, "phrases")))
            if (String(phrase, "text") is { Length: > 0 } text)
            {
                var start = Seconds(String(phrase, "startTime"));
                segments.Add(new() { Text = text, StartSecond = start, EndSecond = start + Seconds(String(phrase, "duration")) });
            }
        var content = AssistantContent(raw);
        return new() { Text = content.ValueKind == JsonValueKind.String ? content.GetString()! : string.Join(' ', segments.Select(x => x.Text)),
            Segments = segments, Language = String(native, "language") ?? language, ProviderMetadata = Metadata(raw),
            Response = new() { ModelId = model, Body = raw, Headers = NativeHttp.SafeResponseMetadataHeaders(upstreamHeaders)
                .Where(pair => pair.Value is not null).ToDictionary(pair => pair.Key, pair => pair.Value!, StringComparer.OrdinalIgnoreCase) } };
    }

    private async Task<(JsonElement Body, Dictionary<string, string?> Headers)> PostMediaAsync(JsonObject body,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        return await NativeHttp.PostJsonAsync(client, "chat/completions", body, ChatAuth(), headers, Json, ct);
    }

    private static JsonObject MediaOptions(IReadOnlyDictionary<string, JsonElement>? options)
    {
        var result = new JsonObject();
        if (options?.TryGetValue("bineric", out var scoped) == true)
        {
            if (scoped.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("bineric options must be an object.");
            foreach (var field in scoped.EnumerateObject()) Add(field.Name, field.Value);
        }
        if (options is not null)
            foreach (var (name, value) in options) Add(name, value);
        return result;

        void Add(string name, JsonElement value)
        {
            if (value.ValueKind != JsonValueKind.Undefined && !MediaControls.Contains(name)) result[name] = Copy(value);
        }
    }

    private static JsonElement AssistantContent(JsonElement raw)
        => Property(Property(Array(Property(raw, "choices")).FirstOrDefault(), "message"), "content");

    private static Dictionary<string, JsonElement> Metadata(JsonElement raw) => new() { ["bineric"] = raw };

    private static float Seconds(string? value)
    {
        if (string.IsNullOrWhiteSpace(value)) return 0;
        try { return (float)XmlConvert.ToTimeSpan(value).TotalSeconds; }
        catch (FormatException) { return 0; }
        catch (OverflowException) { return 0; }
    }
}
