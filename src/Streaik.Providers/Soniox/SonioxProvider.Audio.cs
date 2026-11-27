using System.Net;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class SonioxProvider
{
    private const int AudioLimit = 25 * 1024 * 1024;
    private static readonly HashSet<string> SpeechControls = new(StringComparer.OrdinalIgnoreCase)
        { "model", "input", "text", "instructions", "audio_format", "response_format", "stream_format", "stream" };
    private static readonly HashSet<string> TranscriptionControls = new(StringComparer.OrdinalIgnoreCase)
        { "model", "file", "file_id", "audio", "audio_url", "webhook_url", "webhook_auth_header_name", "webhook_auth_header_value", "stream", "response_format" };

    public Task<HttpResponseMessage> SpeakStreamingAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeakAsync(request, headers, ct);

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var body = Body(request.AdditionalProperties, SpeechControls);
        var parts = request.Model.Split('/', 2);
        var voice = parts.Length == 2 ? parts[1] : request.Voice ?? Text(body["voice"]);
        if (string.IsNullOrWhiteSpace(voice)) throw new ProviderRequestException("Soniox requires a voice or model/voice ID.");
        if (string.IsNullOrWhiteSpace(Text(body["language"]))) throw new ProviderRequestException("Soniox requires a language.");
        if (string.IsNullOrWhiteSpace(request.Input)) throw new ProviderRequestException("Soniox requires speech input.");
        var format = request.ResponseFormat?.Trim().ToLowerInvariant() switch
        {
            null or "" => "mp3", "wave" => "wav", "pcm" or "pcm16" or "raw" => "pcm_s16le", var value => value
        };
        body["model"] = parts[0]; body["voice"] = voice; body["audio_format"] = format; body["text"] = request.Input;
        if (request.Speed is { } speed) body["speed"] = speed;
        using var client = clients.CreateClient(SpeechClient);
        var response = await SendAsync(client, HttpMethod.Post, "tts", JsonContent.Create(body, options: Json), "Bearer", ApiKey(), headers, ct);
        var type = response.Content.Headers.ContentType?.MediaType;
        if (type is null or "application/octet-stream") response.Content.Headers.ContentType = new(format switch
        {
            "mp3" => "audio/mpeg", "wav" => "audio/wav", "pcm_s16le" or "pcm_s16be" => "audio/pcm", _ => "application/octet-stream"
        });
        else if (!type.StartsWith("audio/", StringComparison.OrdinalIgnoreCase))
        {
            response.Dispose();
            throw new IOException("Soniox returned an unexpected speech content type.");
        }
        return response;
    }

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName, string? language,
        IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (audio.CanSeek && (audio.Length - audio.Position is 0 or > AudioLimit))
            throw new ProviderRequestException("Soniox requires a nonempty audio file no larger than 25 MiB.", 422);
        var body = Body(options, TranscriptionControls);
        ValidateTranscriptionOptions(body);
        language ??= Text(body["language"]);
        body.Remove("language");
        if (!string.IsNullOrWhiteSpace(language) && !body.ContainsKey("language_hints")) body["language_hints"] = new JsonArray(language);
        var reference = Text(body["client_reference_id"]);
        var key = ApiKey();
        using var client = clients.CreateClient(Id);
        string? filePath = null;
        string? jobPath = null;
        var completed = false;
        try
        {
            using (var form = new MultipartFormDataContent())
            {
                AddFile(form, "file", audio, mediaType, fileName);
                if (!string.IsNullOrWhiteSpace(reference)) form.Add(new StringContent(reference), "client_reference_id");
                var upload = await JsonAsync<JsonElement>(client, HttpMethod.Post, "v1/files", form, "Bearer", key, headers, ct);
                var fileId = ResourceId(upload);
                filePath = $"v1/files/{Uri.EscapeDataString(fileId)}";
                body["model"] = model;
                body["file_id"] = fileId;
            }
            var started = await JsonAsync<JsonElement>(client, HttpMethod.Post, "v1/transcriptions", JsonContent.Create(body, options: Json), "Bearer", key, headers, ct);
            jobPath = $"v1/transcriptions/{Uri.EscapeDataString(ResourceId(started))}";
            var terminal = await PollAsync(client, key, jobPath, ct);
            var raw = await GetAsync(client, key, jobPath + "/transcript", ct);
            var result = Transcript(model, terminal, raw);
            completed = true;
            return result;
        }
        finally
        {
            // Attempt both deletions even if the request is canceled; a cleanup failure never masks a primary failure.
            var jobFailure = await CleanupAsync(client, key, jobPath);
            var fileFailure = await CleanupAsync(client, key, filePath);
            if (completed && (jobFailure ?? fileFailure) is { } error) throw new IOException("Soniox transcription cleanup failed.", error);
        }
    }

    public async IAsyncEnumerable<StreamingTranscriptionPart> StreamTranscriptionAsync(StreamingTranscriptionRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        if (request.Audio.Length > AudioLimit * 4 / 3 + 8) throw new ProviderRequestException("Soniox audio is too large.", 413);
        byte[] bytes;
        try { bytes = Convert.FromBase64String(request.Audio); }
        catch (FormatException) { throw new ProviderRequestException("Invalid Soniox base64 audio."); }
        var format = request.InputAudioFormat.Type;
        var mime = format.ToLowerInvariant() switch
        {
            "mp3" => "audio/mpeg", "m4a" => "audio/mp4", "opus" => "audio/ogg",
            "wav" or "flac" or "aac" or "ogg" or "webm" => $"audio/{format.ToLowerInvariant()}", _ => format
        };
        if (!mime.StartsWith("audio/", StringComparison.OrdinalIgnoreCase)) throw new ProviderRequestException("Soniox requires an audio MIME type.");
        var extension = mime switch { "audio/mpeg" => "mp3", "audio/x-wav" => "wav", _ => mime[6..] };
        await using var audio = new MemoryStream(bytes, writable: false);
        var result = await TranscribeAsync(request.Model, audio, mime, $"audio.{extension}", null, ScopedOptions(request.ProviderOptions), headers, ct);
        yield return new TranscriptionStreamStartPart();
        yield return new TranscriptionResponseMetadataPart { ModelId = request.Model, Timestamp = result.Response.Timestamp };
        if (request.IncludeRawChunks == true) yield return new TranscriptionRawPart { RawValue = result.ProviderMetadata![Id] };
        if (!string.IsNullOrWhiteSpace(result.Text)) yield return new TranscriptionDeltaPart { Delta = result.Text };
        yield return new TranscriptionFinishPart { Text = result.Text, Language = result.Language, DurationInSeconds = result.DurationInSeconds,
            Segments = result.Segments, ProviderMetadata = result.ProviderMetadata };
    }

    private static JsonObject Body(IReadOnlyDictionary<string, JsonElement>? options, IReadOnlySet<string> reserved)
    {
        var body = new JsonObject();
        foreach (var (name, value) in NativeMediaOptions.Read(Id, options, reserved)) body[name] = Copy(value);
        return body;
    }

    private static string? Text(JsonNode? node) => node is JsonValue value && value.TryGetValue<string>(out var text) ? text : null;

    private static void ValidateTranscriptionOptions(JsonObject body)
    {
        foreach (var (name, value) in body.ToArray())
        {
            var key = name.ToLowerInvariant();
            if (key is not ("language_hints" or "language_hints_strict" or "enable_speaker_diarization" or "enable_language_identification" or "client_reference_id" or "context")) continue;
            if (name != key) { body.Remove(name); body[key] = value; }
            if (value is null) { body.Remove(key); continue; }
            var valid = key switch
            {
                "language_hints" => value is JsonArray hints && hints.All(hint => Text(hint) is not null),
                "client_reference_id" => Text(value) is { Length: <= 256 },
                "context" => value is JsonObject,
                _ => value is JsonValue flag && flag.TryGetValue<bool>(out _)
            };
            if (!valid) throw new ProviderRequestException($"Invalid Soniox {key} option.");
        }
    }

    private static string ResourceId(JsonElement body) => String(body, "id") is { Length: > 0 and <= 256 } id && id is not ("." or "..")
        && id.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_' or '.') ? id : throw new JsonException("Soniox returned an invalid resource ID.");

    private static async Task<JsonElement> PollAsync(HttpClient client, string key, string path, CancellationToken ct)
    {
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromMinutes(5));
        try
        {
            while (true)
            {
                var result = await GetAsync(client, key, path, deadline.Token);
                switch (String(result, "status")?.ToLowerInvariant())
                {
                    case "completed": return result;
                    case "error" or "failed" or "cancelled" or "canceled":
                        throw new IOException(String(result, "error_message") ?? "Soniox transcription failed.");
                    case "queued" or "processing" or "running":
                        await Task.Delay(TimeSpan.FromSeconds(1), deadline.Token); break;
                    default: throw new JsonException("Soniox returned an unrecognized transcription status.");
                }
            }
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested && deadline.IsCancellationRequested)
        { throw new TimeoutException("Soniox transcription job did not complete within five minutes."); }
    }

    private static async Task<Exception?> CleanupAsync(HttpClient client, string key, string? path)
    {
        if (path is null) return null;
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        try
        {
            using var request = new HttpRequestMessage(HttpMethod.Delete, path);
            request.Headers.Authorization = new("Bearer", key);
            using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeout.Token);
            if (response.StatusCode != HttpStatusCode.NotFound) response.EnsureSuccessStatusCode();
            return null;
        }
        catch (Exception error) when (error is HttpRequestException or OperationCanceledException) { return error; }
    }

    private static TranscriptionResponse Transcript(string model, JsonElement terminal, JsonElement raw)
    {
        var tokens = Array(Property(raw, "tokens")).ToArray();
        return new()
        {
            Text = String(raw, "text") ?? string.Concat(tokens.Select(token => String(token, "text"))),
            Language = tokens.Select(token => String(token, "language")).FirstOrDefault(value => !string.IsNullOrWhiteSpace(value)),
            DurationInSeconds = Seconds(terminal, "audio_duration_ms"),
            Segments = tokens.Select(token => new TranscriptionSegment { Text = String(token, "text") ?? "",
                StartSecond = Seconds(token, "start_ms") ?? 0, EndSecond = Seconds(token, "end_ms") ?? 0 }).ToArray(),
            ProviderMetadata = NativeMediaOptions.Metadata(Id, new { transcription = terminal, transcript = raw }),
            Response = new() { ModelId = $"{Id}/{model}", Timestamp = DateTime.UtcNow, Body = raw }
        };
    }

    private static double? Seconds(JsonElement root, string name) => Property(root, name) is { ValueKind: JsonValueKind.Number } value
        && value.TryGetDouble(out var ms) && double.IsFinite(ms) ? ms / 1000 : null;
}
