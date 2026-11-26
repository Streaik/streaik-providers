using System.Globalization;
using System.Net.Http.Headers;
using System.Text.Json;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class CambaiProvider : IStreamingSpeechProvider, ITranscriptionProvider, IChatAudioRoutingProvider
{
    private const int AudioLimit = 32 * 1024 * 1024;
    private static readonly HashSet<string> SpeechControls = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "input", "text", "prompt", "voice", "voice_id", "voice_description", "language", "source_language", "target_language",
        "audio_type", "instructions", "response_format", "stream_format", "stream"
    };
    private static readonly HashSet<string> TranscriptionControls = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "file", "media_file", "language", "stream", "response_format"
    };

    public bool IsSpeechModel(string model) => model.StartsWith("tts" + Source, StringComparison.OrdinalIgnoreCase)
        || model.StartsWith("translated-tts" + Source, StringComparison.OrdinalIgnoreCase)
        || model.ToLowerInvariant() is "text-to-sound/sound" or "text-to-sound/music" or "text-to-voice";

    public bool IsTranscriptionModel(string model) => model.StartsWith("transcribe" + Source, StringComparison.OrdinalIgnoreCase);

    // The upstream returns a finished file; the gateway can frame that file as SSE without presenting it as realtime synthesis.
    public Task<HttpResponseMessage> SpeakStreamingAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeakAsync(request, headers, ct);

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Input)) throw new ProviderRequestException("CAMBAI speech requires text.");
        var native = NativeMediaOptions.Read(Id, request.AdditionalProperties, SpeechControls);
        string operation;
        string fallbackType;
        (string Name, object Value)[] fields;
        if (request.Model.StartsWith("tts" + Source, StringComparison.OrdinalIgnoreCase))
        {
            var (language, _) = LanguageIds(request.Model, "tts");
            operation = "tts";
            fallbackType = "audio/flac";
            fields = [("text", request.Input), ("voice_id", Voice(request.Voice)), ("language", language)];
        }
        else if (request.Model.StartsWith("translated-tts" + Source, StringComparison.OrdinalIgnoreCase))
        {
            var (source, target) = LanguageIds(request.Model, "translated-tts", pair: true);
            operation = "translated-tts";
            fallbackType = "audio/flac";
            fields = [("text", request.Input), ("voice_id", Voice(request.Voice)), ("source_language", source), ("target_language", target!.Value)];
        }
        else if (request.Model.ToLowerInvariant() is "text-to-sound/sound" or "text-to-sound/music")
        {
            operation = "text-to-sound";
            fallbackType = "audio/wav";
            fields = [("prompt", request.Input), ("audio_type", request.Model.EndsWith("/music", StringComparison.OrdinalIgnoreCase) ? "music" : "sound")];
        }
        else if (request.Model.Equals("text-to-voice", StringComparison.OrdinalIgnoreCase))
        {
            if (string.IsNullOrWhiteSpace(request.Instructions)) throw new ProviderRequestException("CAMBAI text-to-voice requires instructions (voice description).");
            operation = "text-to-voice";
            fallbackType = "audio/mpeg";
            fields = [("text", request.Input), ("voice_description", request.Instructions)];
        }
        else throw new ProviderRequestException("Unsupported CAMBAI speech model.", 422);

        using var payload = JobPayload(native, fields);
        var job = await StartJobAsync(operation, payload, ct);
        var resultPath = operation is "tts" or "translated-tts" ? "tts-result" : operation + "-result";
        if (operation == "text-to-voice")
        {
            var result = await JobResultAsync(resultPath, job, ct);
            var preview = Array(Property(result, "previews")).FirstOrDefault(value => value.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(value.GetString()));
            if (preview.ValueKind == JsonValueKind.Undefined) throw new JsonException("CAMBAI returned no voice preview.");
            using var assets = clients.CreateClient(AssetClient);
            using var download = new HttpRequestMessage(HttpMethod.Get, PreviewUri(preview.GetString()!));
            var audio = await ReadAudioAsync(assets, download, fallbackType, ct);
            return new SpeechProviderResponse(audio, NativeMediaOptions.Metadata(Id, job.Raw(result)));
        }

        using var client = clients.CreateClient(Id);
        using var requestAudio = new HttpRequestMessage(HttpMethod.Get, $"{resultPath}/{Number(job.RunId)}");
        requestAudio.Headers.Add("x-api-key", RequiredKey(credentials, Id));
        var content = await ReadAudioAsync(client, requestAudio, fallbackType, ct);
        return new SpeechProviderResponse(content, NativeMediaOptions.Metadata(Id, job.Raw(new
        {
            content_type = content.Headers.ContentType?.MediaType,
            content_length = content.Headers.ContentLength
        })));
    }

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var (id, _) = LanguageIds(model, "transcribe");
        using var form = new MultipartFormDataContent();
        foreach (var (name, value) in NativeMediaOptions.Read(Id, options, TranscriptionControls))
        {
            if (name.Length == 0 || !name.All(c => char.IsAsciiLetterOrDigit(c) || c is '_' or '-' or '[' or ']'))
                throw new ProviderRequestException("Invalid CAMBAI multipart option name.");
            if (value.ValueKind != JsonValueKind.Null)
                form.Add(new StringContent(value.ValueKind == JsonValueKind.String ? value.GetString()! : value.GetRawText()), name);
        }
        form.Add(new StringContent(Number(id)), "language");
        var file = new StreamContent(audio);
        if (MediaTypeHeaderValue.TryParse(mediaType, out var type)) file.Headers.ContentType = type;
        form.Add(file, "media_file", Path.GetFileName(fileName) is { Length: > 0 } nameOnly ? nameOnly : "audio.bin");
        var job = await StartJobAsync("transcribe", form, ct);
        var body = await JobResultAsync("transcription-result", job, ct);
        if (body.ValueKind != JsonValueKind.Array) throw new JsonException("Expected a CAMBAI transcription array.");
        var segments = new List<TranscriptionSegment>();
        foreach (var line in body.EnumerateArray())
        {
            if (String(line, "text") is not { } text || string.IsNullOrWhiteSpace(text)) continue;
            segments.Add(new() { Text = text, StartSecond = Seconds(line, "start"), EndSecond = Seconds(line, "end") });
        }
        return new()
        {
            Text = string.Join(' ', segments.Select(segment => segment.Text)).Trim(), Language = Number(id), Segments = segments,
            DurationInSeconds = segments.Count == 0 ? null : segments.Max(segment => segment.EndSecond),
            Response = new() { ModelId = model, Timestamp = DateTime.UtcNow, Body = job.Raw(body) }
        };
    }

    private Uri PreviewUri(string url)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) || uri.Scheme != Uri.UriSchemeHttps || !uri.IsDefaultPort
            || uri.UserInfo.Length != 0 || uri.Fragment.Length != 0 || uri.HostNameType != UriHostNameType.Dns)
            throw new JsonException("Invalid CAMBAI preview URL.");
        var host = uri.IdnHost;
        if (!host.Equals("camb.ai", StringComparison.OrdinalIgnoreCase) && !host.EndsWith(".camb.ai", StringComparison.OrdinalIgnoreCase)
            && !string.Equals(host, configuration["Providers:cambai:TrustedPreviewHost"], StringComparison.OrdinalIgnoreCase))
            throw new JsonException("Untrusted CAMBAI preview host.");
        return uri;
    }

    private static async Task<ByteArrayContent> ReadAudioAsync(HttpClient client, HttpRequestMessage request, string fallbackType, CancellationToken ct)
    {
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromMinutes(2));
        var token = deadline.Token;
        using var response = await NativeHttp.SendAsync(client, request, token);
        var mediaType = response.Content.Headers.ContentType?.MediaType;
        if (mediaType is null or "application/octet-stream" or "binary/octet-stream") mediaType = fallbackType;
        if (!mediaType.StartsWith("audio/", StringComparison.OrdinalIgnoreCase)) throw new JsonException("CAMBAI returned a non-audio result.");
        if (response.Content.Headers.ContentLength > AudioLimit) throw new JsonException("CAMBAI audio is too large.");
        await using var stream = await response.Content.ReadAsStreamAsync(token);
        using var buffer = new MemoryStream();
        var chunk = new byte[16_384];
        while (await stream.ReadAsync(chunk, token) is > 0 and var length)
        {
            if (buffer.Length + length > AudioLimit) throw new JsonException("CAMBAI audio is too large.");
            buffer.Write(chunk, 0, length);
        }
        if (buffer.Length == 0) throw new JsonException("CAMBAI returned no audio.");
        return new ByteArrayContent(buffer.GetBuffer(), 0, (int)buffer.Length) { Headers = { ContentType = new(mediaType), ContentLength = buffer.Length } };
    }

    private static int Voice(string? voice)
        => string.IsNullOrWhiteSpace(voice) ? 147320
            : int.TryParse(voice, NumberStyles.None, CultureInfo.InvariantCulture, out var id) && id > 0 ? id
            : throw new ProviderRequestException("CAMBAI voice must be a positive integer voice ID.");

    private static double Seconds(JsonElement value, string name)
        => Property(value, name) is { ValueKind: JsonValueKind.Number } number && number.TryGetDouble(out var seconds) && double.IsFinite(seconds)
            ? seconds : 0;
}
