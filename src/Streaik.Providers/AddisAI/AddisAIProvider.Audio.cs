using System.Text.Json;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class AddisAIProvider : IStreamingSpeechProvider, ITranscriptionProvider, IChatAudioRoutingProvider
{
    private const int MaxAudioBytes = 32 * 1024 * 1024;
    private static readonly HashSet<string> SpeechControls = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "input", "text", "language", "voice", "instructions", "speed", "response_format", "stream", "stream_format"
    };
    private static readonly HashSet<string> TranscriptionControls = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "file", "audio", "language", "language_code", "response_format", "stream", "request_data"
    };

    public bool IsSpeechModel(string model) => model.StartsWith("tts/", StringComparison.OrdinalIgnoreCase);
    public bool IsTranscriptionModel(string model) => model.StartsWith("stt/", StringComparison.OrdinalIgnoreCase);

    // Stream the completed WAV through the gateway's existing binary-to-SSE adapter.
    public Task<HttpResponseMessage> SpeakStreamingAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeakAsync(request, headers, ct);

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Input)) throw new ProviderRequestException("AddisAI speech requires text.");
        var language = AudioLanguage(request.Model, "tts");
        var payload = Options(request.AdditionalProperties, SpeechControls);
        payload["text"] = request.Input;
        payload["language"] = language;
        payload["stream"] = false;
        var key = RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        var raw = await PostAsync(client, "v1/audio", payload, key, headers, ct);
        var encoded = RequiredText(raw, "audio");
        if (encoded.Length > (MaxAudioBytes + 2) / 3 * 4) throw new JsonException("AddisAI audio is too large.");
        byte[] bytes;
        try { bytes = Convert.FromBase64String(encoded); }
        catch (FormatException ex) { throw new JsonException("AddisAI returned invalid base64 audio.", ex); }
        if (bytes.Length == 0 || bytes.Length > MaxAudioBytes) throw new JsonException("AddisAI returned an invalid audio length.");
        var metadata = raw.EnumerateObject().Where(field => !field.NameEquals("audio")).ToDictionary(field => field.Name, field => field.Value);
        return new SpeechProviderResponse(new ByteArrayContent(bytes) { Headers = { ContentType = new("audio/wav") } },
            NativeMediaOptions.Metadata(Id, new { language, response = metadata }));
    }

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var selectedLanguage = AudioLanguage(model, "stt");
        var requestData = Options(options, TranscriptionControls);
        requestData["language_code"] = selectedLanguage;
        var key = RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        using var form = new MultipartFormDataContent();
        AddFile(form, "audio", audio, mediaType, fileName);
        form.Add(new StringContent(JsonSerializer.Serialize(requestData, Json)), "request_data");
        using var request = new HttpRequestMessage(HttpMethod.Post, "v2/stt") { Content = form };
        request.Headers.TryAddWithoutValidation("X-API-Key", key);
        NativeHttp.ApplyHeaders(request, headers);
        using var response = await NativeHttp.SendAsync(client, request, ct);
        var raw = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        var native = new Dictionary<string, object?> { ["language"] = selectedLanguage, ["response"] = raw };
        if (raw.ValueKind == JsonValueKind.Object && raw.TryGetProperty("confidence", out var confidence)) native["confidence"] = confidence;
        return new()
        {
            Text = RequiredText(Property(raw, "data"), "transcription"), Language = selectedLanguage,
            ProviderMetadata = NativeMediaOptions.Metadata(Id, native),
            Response = new() { Body = raw }
        };
    }

    private static string AudioLanguage(string model, string mode) => model.ToLowerInvariant().Split('/') switch
    {
        [var prefix, var language] when prefix == mode && language is "am" or "om" => language,
        _ => throw new ProviderRequestException($"Use an AddisAI {mode}/am or {mode}/om model.", 422)
    };
}
