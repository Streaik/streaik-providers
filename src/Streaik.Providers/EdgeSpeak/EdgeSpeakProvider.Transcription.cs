using System.Globalization;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class EdgeSpeakProvider
{
    private const string TranscriptionPath = "audio/transcriptions";

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName, string? language,
        IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var fields = new Dictionary<string, JsonElement>(options ?? new Dictionary<string, JsonElement>(), StringComparer.Ordinal);
        if (!string.IsNullOrWhiteSpace(language)) fields["language"] = Value(language);
        var format = OptionString(fields, "response_format") ?? "verbose_json";
        using var upstream = await SendTranscriptionAsync(model, audio, mediaType, fileName, fields, format, false, headers, ct);
        var body = format == "text" ? Value(new { text = await upstream.Content.ReadAsStringAsync(ct) })
            : await NativeHttp.ReadAsync<JsonElement>(upstream, Json, ct);
        return new()
        {
            Text = RequiredString(body, "text"), Language = String(body, "language"), DurationInSeconds = (float?)Number(body, "duration"),
            Segments = Segments(body), ProviderMetadata = Metadata(body),
            Response = new() { ModelId = model, Timestamp = DateTime.UtcNow, Headers = ResponseHeaders(upstream), Body = body }
        };
    }

    public async Task<HttpResponseMessage> TranscribeNativeAsync(string model, IFormCollection request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Keys.Any(key => key is "chunking_strategy" or "include" or "include[]" or "known_speaker_names" or "known_speaker_names[]"
            or "known_speaker_references" or "known_speaker_references[]"))
            throw new ProviderRequestException("EdgeSpeak does not support chunking, include, or known speaker references.");
        var fields = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (var (key, values) in request)
        {
            if (key is "file" or "model" or "stream") continue;
            if (key is "timestamp_granularities" or "timestamp_granularities[]")
            {
                fields[key] = values.Count == 1 && values[0]?.StartsWith('[') == true ? ParseField(values[0]) : Value(values.ToArray());
                continue;
            }
            fields[key] = key is "candidates" or "word_timestamp_enabled" or "semantic_sentence_enabled"
                ? ParseField(values.ToString()) : Value(values.ToString());
        }
        var format = OptionString(fields, "response_format");
        if (string.IsNullOrEmpty(format)) format = "json";
        var rawStream = request["stream"].ToString();
        if (rawStream != "" && !bool.TryParse(rawStream, out _)) throw new ProviderRequestException("stream must be true or false.");
        var streaming = string.Equals(rawStream, "true", StringComparison.OrdinalIgnoreCase);
        var file = request.Files.GetFile("file") ?? throw new ProviderRequestException("EdgeSpeak requires an audio file.");
        await using var audio = file.OpenReadStream();
        var upstream = await SendTranscriptionAsync(model, audio, file.ContentType, file.FileName, fields, format, streaming, headers, ct);
        if (streaming || format == "text") return upstream;
        using (upstream)
        {
            var raw = await NativeHttp.ReadAsync<JsonElement>(upstream, Json, ct);
            return new(System.Net.HttpStatusCode.OK) { Content = JsonContent.Create(NativeTranscriptionBody(raw, format), options: Json) };
        }
    }

    private static JsonObject NativeTranscriptionBody(JsonElement raw, string format)
    {
        _ = RequiredString(raw, "text");
        var body = (JsonObject)Copy(raw)!;
        if (body["usage"] is JsonObject usage && usage["type"] is null && Number(Property(raw, "usage"), "seconds") is not null)
            usage["type"] = "duration";
        if (format == "json") return body;

        // Normalize required OpenAI fields without discarding EdgeSpeak's diagnostics or candidate data.
        body["task"] = "transcribe";
        body["duration"] = Number(raw, "duration") ?? throw new JsonException("EdgeSpeak transcription is missing duration.");
        if (format == "verbose_json") body["language"] = RequiredString(raw, "language");
        if (body["segments"] is not JsonArray segments) body["segments"] = segments = new JsonArray();
        for (var i = 0; i < segments.Count; i++)
        {
            if (segments[i] is not JsonObject segment) throw new JsonException("Invalid EdgeSpeak transcription segment.");
            if (format == "diarized_json")
            {
                segment["type"] = "transcript.text.segment";
                segment["id"] = segment["id"]?.ToString() ?? i.ToString(CultureInfo.InvariantCulture);
                segment["speaker"] ??= "unknown";
                continue;
            }
            if (segment["id"] is not JsonValue id || !id.TryGetValue<int>(out _)) segment["id"] = i;
            segment["seek"] ??= 0;
            segment["tokens"] ??= new JsonArray();
            segment["temperature"] ??= 0;
            segment["avg_logprob"] ??= 0;
            segment["compression_ratio"] ??= 0;
            segment["no_speech_prob"] ??= 0;
        }
        if (format == "verbose_json" && body["words"] is null)
        {
            var words = segments.OfType<JsonObject>().SelectMany(segment => segment["words"] as JsonArray ?? [])
                .Select(word => word?.DeepClone()).ToArray();
            if (words.Length > 0) body["words"] = new JsonArray(words);
        }
        return body;
    }

    public async IAsyncEnumerable<StreamingTranscriptionPart> StreamTranscriptionAsync(StreamingTranscriptionRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        if (request.InputAudioFormat?.Rate is <= 0) throw new ProviderRequestException("EdgeSpeak audio rate must be positive.");
        var mediaType = request.InputAudioFormat?.Type;
        var encoded = request.Audio;
        if (encoded.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
        {
            var comma = encoded.IndexOf(',');
            if (comma < 0 || !encoded[..comma].EndsWith(";base64", StringComparison.OrdinalIgnoreCase))
                throw new ProviderRequestException("EdgeSpeak requires a base64 audio data URL.");
            encoded = encoded[(comma + 1)..];
        }
        byte[] bytes;
        try { bytes = Convert.FromBase64String(encoded); }
        catch (FormatException) { throw new ProviderRequestException("Invalid EdgeSpeak base64 audio."); }
        if (bytes.Length == 0) throw new ProviderRequestException("EdgeSpeak requires nonempty audio.");
        using var audio = new MemoryStream(bytes, writable: false);
        var fields = ScopedOptions(request.ProviderOptions);
        using var upstream = await SendTranscriptionAsync(request.Model, audio, mediaType, null, fields,
            OptionString(fields, "response_format") ?? "json", true, headers, ct);
        yield return new TranscriptionStreamStartPart();
        yield return new TranscriptionResponseMetadataPart { Timestamp = DateTime.UtcNow, ModelId = request.Model, Headers = ResponseHeaders(upstream) };
        await foreach (var (type, data) in Events(upstream, "transcript.text.done", ct))
        {
            if (request.IncludeRawChunks == true) yield return new TranscriptionRawPart { RawValue = RawChunk(type, data) };
            if (type == "transcript.text.delta")
                yield return new TranscriptionDeltaPart { Delta = RequiredString(data, "delta"), ProviderMetadata = Metadata(data) };
            else if (type == "transcript.text.done")
                yield return new TranscriptionFinishPart { Text = RequiredString(data, "text"), Language = String(data, "language"),
                    DurationInSeconds = Number(data, "duration"), Segments = Segments(data), ProviderMetadata = Metadata(data) };
        }
    }

    private async Task<HttpResponseMessage> SendTranscriptionAsync(string model, Stream audio, string? mediaType, string? fileName,
        Dictionary<string, JsonElement> fields, string format, bool streaming, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (format is not ("json" or "verbose_json" or "diarized_json" or "text"))
            throw new ProviderRequestException("Unsupported EdgeSpeak transcription response_format.");
        var granularities = TimestampGranularities(fields);
        ValidateTranscription(fields, granularities, format, streaming);
        if (string.IsNullOrWhiteSpace(mediaType)) throw new ProviderRequestException("EdgeSpeak requires an audio media type.");
        if (!mediaType.Contains('/')) mediaType = "audio/" + mediaType;
        mediaType = mediaType.ToLowerInvariant() switch { "audio/mp3" => "audio/mpeg", "audio/wave" => "audio/wav", _ => mediaType };
        if (!MediaTypeHeaderValue.TryParse(mediaType, out var type)) throw new ProviderRequestException("Invalid EdgeSpeak audio media type.");
        var extension = type.MediaType switch
        {
            "audio/wav" or "audio/x-wav" => "wav", "audio/mpeg" => "mp3", "audio/flac" => "flac", "audio/ogg" => "ogg",
            "audio/opus" => "opus", "audio/aac" => "aac", "audio/mp4" or "video/mp4" => "mp4", "audio/webm" or "video/webm" => "webm", _ => "bin"
        };
        using var form = new MultipartFormDataContent();
        var file = new StreamContent(audio);
        file.Headers.ContentType = type;
        form.Add(file, "file", string.IsNullOrEmpty(Path.GetExtension(fileName)) ? $"audio.{extension}" : fileName!);
        form.Add(new StringContent(model), "model");
        form.Add(new StringContent(format), "response_format");
        form.Add(new StringContent(streaming ? "true" : "false"), "stream");
        foreach (var granularity in granularities) form.Add(new StringContent(granularity), "timestamp_granularities[]");
        foreach (var (key, value) in fields)
            if (key.ToLowerInvariant() is not ("file" or "model" or "stream" or "response_format" or "timestamp_granularities" or "timestamp_granularities[]")
                && value.ValueKind != JsonValueKind.Null)
                form.Add(new StringContent(value.ValueKind == JsonValueKind.String ? value.GetString()! : value.GetRawText()), key);
        var accept = streaming ? "text/event-stream" : format == "text" ? "text/plain" : "application/json";
        var upstream = await SendMediaAsync(TranscriptionPath, form, accept, headers, ct);
        if (upstream.Content.Headers.ContentType?.MediaType == accept) return upstream;
        upstream.Dispose();
        throw new IOException($"EdgeSpeak did not return {accept} transcription output.");
    }

    private static string[] TimestampGranularities(Dictionary<string, JsonElement> fields)
    {
        var result = new List<string>();
        foreach (var name in new[] { "timestamp_granularities", "timestamp_granularities[]" })
        {
            if (!fields.TryGetValue(name, out var values) || values.ValueKind == JsonValueKind.Null) continue;
            if (values.ValueKind != JsonValueKind.Array) throw new ProviderRequestException("Timestamp granularities must be an array.");
            foreach (var value in values.EnumerateArray())
            {
                if (value.ValueKind != JsonValueKind.String || value.GetString() is not ("word" or "segment"))
                    throw new ProviderRequestException("Timestamp granularities must be word or segment.");
                result.Add(value.GetString()!);
            }
        }
        if (result.Count > 8) throw new ProviderRequestException("EdgeSpeak accepts at most eight timestamp granularities.");
        return result.ToArray();
    }

    private static void ValidateTranscription(Dictionary<string, JsonElement> fields, string[] granularities, string format, bool streaming)
    {
        bool Has(string key) => fields.TryGetValue(key, out var value) && value.ValueKind != JsonValueKind.Null;
        var diarization = Has("diarization_model");
        if (streaming && (format != "json" || diarization || granularities.Length > 0))
            throw new ProviderRequestException("EdgeSpeak streaming transcription requires json without diarization or timestamp granularities.");
        if (diarization && format != "diarized_json" || granularities.Length > 0 && format is not ("verbose_json" or "diarized_json"))
            throw new ProviderRequestException("Use a detailed format for EdgeSpeak diarization and timestamps.");
        var wordTiming = fields.GetValueOrDefault("word_timestamp_enabled").ValueKind;
        if (wordTiming is not (JsonValueKind.Undefined or JsonValueKind.Null or JsonValueKind.True or JsonValueKind.False)
            || wordTiming == JsonValueKind.True && format is not ("verbose_json" or "diarized_json")
            || wordTiming == JsonValueKind.False && granularities.Contains("word"))
            throw new ProviderRequestException("Incompatible EdgeSpeak word timestamp settings.");
        if (Has("candidates") && (fields["candidates"].ValueKind != JsonValueKind.Number || !fields["candidates"].TryGetInt32(out var count)
            || count is < 0 or > 8 || count >= 2 && (format != "verbose_json" || granularities.Contains("word") || wordTiming == JsonValueKind.True)))
            throw new ProviderRequestException("EdgeSpeak alternatives require zero to eight candidates and verbose_json without word timestamps for multiple candidates.");
        if (fields.GetValueOrDefault("semantic_sentence_enabled").ValueKind == JsonValueKind.False &&
            new[] { "min_chars", "max_chars", "start_margin", "end_margin" }.Any(Has))
            throw new ProviderRequestException("EdgeSpeak sentence controls require semantic sentences.");
    }

    private static JsonElement ParseField(string? value)
    {
        try { using var doc = JsonDocument.Parse(value ?? "null"); return doc.RootElement.Clone(); }
        catch (JsonException) { throw new ProviderRequestException("Invalid EdgeSpeak transcription form option."); }
    }

    private static TranscriptionSegment[] Segments(JsonElement body)
        => Array(Property(body, "segments")).Select(segment => new TranscriptionSegment
        {
            Text = RequiredString(segment, "text"),
            StartSecond = (float)(Number(segment, "start") ?? throw new JsonException("EdgeSpeak segment is missing its start.")),
            EndSecond = (float)(Number(segment, "end") ?? throw new JsonException("EdgeSpeak segment is missing its end."))
        }).ToArray();
}
