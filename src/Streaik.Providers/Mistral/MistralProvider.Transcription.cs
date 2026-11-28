using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Mistral;

internal sealed partial class MistralProvider
{
    private const string TranscriptionPath = "audio/transcriptions";
    private const string TranscriptionDone = "transcription.done";

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var fields = AudioOptions(options);
        if (!string.IsNullOrWhiteSpace(language)) fields["language"] = JsonSerializer.SerializeToElement(language);
        using var client = CreateClient();
        using var message = new HttpRequestMessage(HttpMethod.Post, TranscriptionPath)
        { Content = TranscriptionForm(model, audio, mediaType, fileName, fields, false, []) };
        using var response = await SendAudioAsync(client, message, headers, "application/json", ct);
        var payload = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        var finish = FinishTranscription(payload, [], language);
        return new() { Text = finish.Text, Segments = finish.Segments, Language = finish.Language,
            DurationInSeconds = finish.DurationInSeconds, ProviderMetadata = finish.ProviderMetadata,
            Response = new() { Timestamp = DateTime.UtcNow, ModelId = String(payload, "model") ?? model, Body = payload } };
    }

    public async IAsyncEnumerable<StreamingTranscriptionPart> StreamTranscriptionAsync(StreamingTranscriptionRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        var bytes = AudioBytes(request.Audio, upstream: false);
        using var audio = new MemoryStream(bytes, writable: false);
        var media = request.InputAudioFormat.Type;
        var warnings = new List<object>();
        var options = AudioOptions(request.ProviderOptions, namespaced: true);
        using var client = CreateClient();
        using var message = new HttpRequestMessage(HttpMethod.Post, TranscriptionPath)
        { Content = TranscriptionForm(request.Model, audio, media, AudioFileName(media), options, true, warnings) };
        using var response = await SendAudioAsync(client, message, headers, "text/event-stream", ct);
        yield return new TranscriptionStreamStartPart { Warnings = warnings };
        yield return new TranscriptionResponseMetadataPart { Timestamp = DateTime.UtcNow, ModelId = request.Model };
        var segments = new List<JsonElement>();
        var language = options.GetValueOrDefault("language") is { ValueKind: JsonValueKind.String } value ? value.GetString() : null;
        await foreach (var (type, payload) in AudioEventsAsync(response, TranscriptionDone, ct))
        {
            if (request.IncludeRawChunks == true) yield return new TranscriptionRawPart { RawValue = payload };
            switch (type)
            {
                case "transcription.language": language = String(payload, "language"); break;
                case "transcription.text.delta": yield return new TranscriptionDeltaPart { Delta = RequiredString(payload, "text") }; break;
                case "transcription.segment.delta":
                    segments.Add(payload);
                    yield return new TranscriptionFinalPart { Text = RequiredString(payload, "text"), StartSecond = AudioDouble(payload, "start"),
                        EndSecond = AudioDouble(payload, "end"), ProviderMetadata = AudioMetadata(payload) };
                    break;
                case TranscriptionDone: yield return FinishTranscription(payload, segments, language); break;
            }
        }
    }

    public async Task<HttpResponseMessage> TranscribeNativeAsync(string model, IFormCollection request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Files.GetFile("file") is not { Length: > 0 } file) throw new ProviderRequestException("An audio file is required.");
        var format = request["response_format"].ToString();
        var stream = request["stream"] == "true";
        if (format is not ("" or "json" or "text" or "verbose_json") || stream && format is "text" or "verbose_json")
            throw new ProviderRequestException("Unsupported Mistral transcription response format.");
        var options = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (var (raw, values) in request)
        {
            var name = raw.TrimEnd('[', ']');
            if (name is "timestamp_granularities" or "context_bias")
                options[name] = JsonSerializer.SerializeToElement(values.ToArray());
            else if (name is "diarize" or "return_language" && bool.TryParse(values.ToString(), out var boolean))
                options[name] = JsonSerializer.SerializeToElement(boolean);
            else if (name == "temperature" && double.TryParse(values.ToString(), System.Globalization.CultureInfo.InvariantCulture, out var temperature)
                     && double.IsFinite(temperature))
                options[name] = JsonSerializer.SerializeToElement(temperature);
            else if (name is "language" or "temperature" or "diarize" or "return_language")
                options[name] = JsonSerializer.SerializeToElement(values.ToString());
        }
        using var audio = file.OpenReadStream();
        if (!stream)
        {
            var result = await TranscribeAsync(model, audio, file.ContentType, file.FileName, null, options, headers, ct);
            return new(HttpStatusCode.OK) { Content = format == "text" ? new StringContent(result.Text, Encoding.UTF8, "text/plain")
                : JsonContent.Create(result.Response?.Body ?? new { text = result.Text }) };
        }
        using var client = CreateClient();
        using var message = new HttpRequestMessage(HttpMethod.Post, TranscriptionPath)
        { Content = TranscriptionForm(model, audio, file.ContentType, file.FileName, options, true, []) };
        var upstream = await SendAudioAsync(client, message, headers, "text/event-stream", ct);
        if (upstream.Content.Headers.ContentType?.MediaType != "text/event-stream")
        {
            upstream.Dispose();
            throw new IOException("Expected Mistral transcription event stream.");
        }
        return new(HttpStatusCode.OK) { Content = new NativeTranscriptionStream(upstream) };
    }

    private static MultipartFormDataContent TranscriptionForm(string model, Stream audio, string mediaType, string fileName,
        Dictionary<string, JsonElement> options, bool stream, List<object> warnings)
    {
        if (IsRealtimeTranscription(model)) throw new ProviderRequestException("Use a Mistral realtime token for this model.");
        if (!MediaTypeHeaderValue.TryParse(mediaType, out var contentType)
            || contentType.MediaType is not { } type || !(type.StartsWith("audio/", StringComparison.OrdinalIgnoreCase) || type == "video/webm" || type == "video/mp4"))
            throw new ProviderRequestException("Mistral transcription requires an audio MIME type.");
        var form = new MultipartFormDataContent();
        try
        {
            var file = new StreamContent(audio);
            file.Headers.ContentType = contentType;
            form.Add(file, "file", string.IsNullOrWhiteSpace(fileName) ? AudioFileName(type) : fileName);
            form.Add(new StringContent(model), "model");
            if (stream) form.Add(new StringContent("true"), "stream");
            foreach (var (name, value) in options)
            {
                if (value.ValueKind == JsonValueKind.Null || name is "file" or "file_url" or "file_id" or "stream" or "model") continue;
                if (name is "timestamp_granularities" or "context_bias")
                {
                    if (value.ValueKind != JsonValueKind.Array || value.EnumerateArray().Any(item => item.ValueKind != JsonValueKind.String))
                        throw new ProviderRequestException($"Mistral transcription option '{name}' must be an array of strings.");
                    foreach (var entry in value.EnumerateArray()) form.Add(new StringContent(entry.GetString()!), name);
                }
                else if (name is "language" or "temperature" or "diarize" or "return_language")
                {
                    var valid = name switch { "language" => value.ValueKind == JsonValueKind.String,
                        "temperature" => value.ValueKind == JsonValueKind.Number, _ => value.ValueKind is JsonValueKind.True or JsonValueKind.False };
                    if (!valid) throw new ProviderRequestException($"Mistral transcription option '{name}' has an invalid type.");
                    form.Add(new StringContent(value.ValueKind == JsonValueKind.String ? value.GetString()! : value.GetRawText()), name);
                }
                else warnings.Add(new { type = "unsupported", feature = "providerOptions.mistral." + name });
            }
            return form;
        }
        catch { form.Dispose(); throw; }
    }

    private static TranscriptionFinishPart FinishTranscription(JsonElement payload, List<JsonElement> streamed, string? language)
    {
        var rawSegments = Property(payload, "segments");
        var segments = rawSegments.ValueKind switch
        {
            JsonValueKind.Array => rawSegments.EnumerateArray().ToList(),
            JsonValueKind.Null or JsonValueKind.Undefined => streamed,
            _ => throw new JsonException("Mistral returned invalid transcription segments.")
        };
        var mapped = new List<TranscriptionSegment>();
        foreach (var segment in segments)
        {
            var text = RequiredString(segment, "text");
            if (AudioDouble(segment, "start") is double start && Math.Abs(start) <= float.MaxValue
                && AudioDouble(segment, "end") is double end && Math.Abs(end) <= float.MaxValue)
                mapped.Add(new() { Text = text, StartSecond = (float)start, EndSecond = (float)end });
        }
        var raw = payload.EnumerateObject().ToDictionary(x => x.Name, x => x.Value);
        if (rawSegments.ValueKind == JsonValueKind.Undefined) raw["segments"] = JsonSerializer.SerializeToElement(segments);
        else if (rawSegments.ValueKind == JsonValueKind.Null && streamed.Count > 0) raw["streamed_segments"] = JsonSerializer.SerializeToElement(streamed);
        var usage = Property(payload, "usage");
        return new() { Text = RequiredString(payload, "text"), Segments = mapped,
            Language = Property(payload, "language").ValueKind == JsonValueKind.Undefined ? language : String(payload, "language"),
            DurationInSeconds = AudioDouble(usage, "prompt_audio_seconds"), ProviderMetadata = AudioMetadata(JsonSerializer.SerializeToElement(raw)),
            Usage = usage.ValueKind == JsonValueKind.Object ? usage.EnumerateObject().ToDictionary(x => x.Name, x => x.Value) : null };
    }

    private static string AudioFileName(string media) => "audio." + (media.ToLowerInvariant() switch
    { "audio/mpeg" or "audio/mp3" => "mp3", "audio/wav" or "audio/x-wav" => "wav", "audio/mp4" or "video/mp4" => "mp4",
      "audio/webm" or "video/webm" => "webm", "audio/ogg" => "ogg", "audio/flac" => "flac", "audio/aac" => "aac", "audio/opus" => "opus", _ => "bin" });

    private sealed class NativeTranscriptionStream : HttpContent
    {
        private readonly HttpResponseMessage upstream;
        public NativeTranscriptionStream(HttpResponseMessage upstream)
        { this.upstream = upstream; Headers.ContentType = new("text/event-stream"); }

        protected override bool TryComputeLength(out long length) { length = 0; return false; }
        protected override Task SerializeToStreamAsync(Stream stream, TransportContext? context)
            => SerializeToStreamAsync(stream, context, CancellationToken.None);
        protected override async Task SerializeToStreamAsync(Stream stream, TransportContext? context, CancellationToken ct)
        {
            await foreach (var (type, payload) in AudioEventsAsync(upstream, TranscriptionDone, ct))
            {
                object? part = type switch
                {
                    "transcription.text.delta" => new { type = "transcript.text.delta", delta = RequiredString(payload, "text") },
                    TranscriptionDone => new { type = "transcript.text.done", text = RequiredString(payload, "text") },
                    _ => null
                };
                if (part is null) continue;
                await stream.WriteAsync(Encoding.UTF8.GetBytes($"data: {JsonSerializer.Serialize(part)}\n\n"), ct);
                await stream.FlushAsync(ct);
            }
        }
        protected override void Dispose(bool disposing) { if (disposing) upstream.Dispose(); base.Dispose(disposing); }
    }
}
