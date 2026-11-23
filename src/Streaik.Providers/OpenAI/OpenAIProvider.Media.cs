using System.Net;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.OpenAI;

public sealed partial class OpenAIProvider
{
    public Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PostAsync<OpenAIEmbeddingRequest, OpenAIEmbeddingResponse>("v1/embeddings", request, headers, Json, ct);

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PostAsync<OpenAIImageGenerationRequest, OpenAIImagesResponse>("v1/images/generations", request, headers, Json, ct);

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => EditImageCoreAsync(request, headers, ct);

    private async Task<OpenAIImagesResponse> EditImageCoreAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var key = RequireKey();
        using var client = clients.CreateClient(Id);
        using var body = EditPayload(request);
        return await JsonAsync<OpenAIImagesResponse>(client, HttpMethod.Post, "v1/images/edits", body, "Bearer", key, headers, ct);
    }

    public IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamAsync<OpenAIImageGenerationRequest, JsonElement>("v1/images/generations", request, headers, Json, ImageTerminal, ct);

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var key = RequireKey();
        using var client = clients.CreateClient(Id);
        using var body = EditPayload(request);
        using var response = await SendAsync(client, HttpMethod.Post, "v1/images/edits", body, "Bearer", key, headers, ct);
        await foreach (var part in NativeHttp.ReadSseAsync<JsonElement>(response, Json, ImageTerminal, ct)) yield return part;
    }

    private static bool ImageTerminal(JsonElement part) => String(part, "type") is "image_generation.completed" or "image_edit.completed" or "error";

    private static HttpContent EditPayload(OpenAIImageEditRequest request)
    {
        if (request.ImageFiles is not { Length: > 0 } && request.MaskFile is null) return JsonContent.Create(request, options: Json);
        var form = ImageMultipart(request, inlineReferences: false, "image[]", "mask");
        if (request.Stream is { } stream) form.Add(new StringContent(stream ? "true" : "false"), "stream");
        if (request.PartialImages is { } partial) form.Add(new StringContent(partial.ToString(System.Globalization.CultureInfo.InvariantCulture)), "partial_images");
        return form;
    }

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var key = RequireKey();
        using var client = clients.CreateClient(Id);
        return await SendAsync(client, HttpMethod.Post, "v1/audio/speech", JsonContent.Create(request, options: Json), "Bearer", key, headers, ct);
    }

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName, string? language,
        IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var key = RequireKey();
        using var client = clients.CreateClient(Id);
        using var form = new MultipartFormDataContent();
        if (options is not null)
            foreach (var item in options.Where(x => x.Key is not ("file" or "model" or "stream" or "language" or "response_format")))
                form.Add(new StringContent(item.Value.ValueKind == JsonValueKind.String ? item.Value.GetString()! : item.Value.GetRawText()), item.Key);
        form.Add(new StringContent(model), "model");
        form.Add(new StringContent(model == "whisper-1" ? "verbose_json" : "json"), "response_format");
        if (!string.IsNullOrWhiteSpace(language)) form.Add(new StringContent(language), "language");
        AddFile(form, "file", audio, mediaType, fileName);
        var root = await JsonAsync<JsonElement>(client, HttpMethod.Post, "v1/audio/transcriptions", form, "Bearer", key, headers, ct);
        var segments = Array(Property(root, "segments")).Select(x => new TranscriptionSegment { Text = String(x, "text") ?? "",
            StartSecond = Number(x, "start") ?? 0, EndSecond = Number(x, "end") ?? 0 });
        return new() { Text = String(root, "text") ?? throw new JsonException("Missing transcription text."), Language = String(root, "language"),
            DurationInSeconds = Number(root, "duration"), Segments = segments.ToArray() };
    }

    public async Task<HttpResponseMessage> TranscribeNativeAsync(string model, IFormCollection request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var key = RequireKey();
        using var client = clients.CreateClient(Id);
        using var form = TranscriptionForm(model);
        foreach (var (name, values) in request)
            if (name is not ("model" or "file"))
                foreach (var value in values) form.Add(new StringContent(value ?? ""), name);
        var file = request.Files.GetFile("file") ?? throw new ProviderRequestException("'file' is required.");
        AddFile(form, "file", file);
        return await SendAsync(client, HttpMethod.Post, "v1/audio/transcriptions", form, "Bearer", key, headers, ct);
    }

    public async IAsyncEnumerable<StreamingTranscriptionPart> StreamTranscriptionAsync(StreamingTranscriptionRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        var (extension, mediaType) = request.InputAudioFormat.Type switch
        {
            "mp3" or "audio/mpeg" => ("mp3", "audio/mpeg"), "wav" or "audio/wav" => ("wav", "audio/wav"),
            "webm" or "audio/webm" => ("webm", "audio/webm"), "flac" or "audio/flac" => ("flac", "audio/flac"),
            "mp4" or "audio/mp4" or "m4a" => ("m4a", "audio/mp4"), _ => throw new ProviderRequestException("Unsupported OpenAI transcription audio format.", 415)
        };
        byte[] bytes;
        try { bytes = Convert.FromBase64String(request.Audio); }
        catch (FormatException) { throw new ProviderRequestException("Invalid base64 audio."); }
        if (bytes.Length is 0 or > 25 * 1024 * 1024) throw new ProviderRequestException("Invalid audio length.", 413);
        var key = RequireKey();
        using var client = clients.CreateClient(Id);
        using var form = TranscriptionForm(request.Model);
        if (request.ProviderOptions?.TryGetValue(Id, out var options) == true && options.ValueKind == JsonValueKind.Object)
            foreach (var option in options.EnumerateObject().Where(p => p.Name is "language" or "prompt" or "temperature"))
                form.Add(new StringContent(option.Value.ValueKind == JsonValueKind.String ? option.Value.GetString()! : option.Value.GetRawText()), option.Name);
        form.Add(new StringContent("true"), "stream");
        AddFile(form, "file", new MemoryStream(bytes, writable: false), mediaType, $"audio.{extension}");
        using var response = await SendAsync(client, HttpMethod.Post, "v1/audio/transcriptions", form, "Bearer", key, headers, ct);
        yield return new TranscriptionStreamStartPart();
        yield return new TranscriptionResponseMetadataPart { Timestamp = DateTime.UtcNow, ModelId = request.Model };
        await foreach (var part in NativeHttp.ReadSseAsync<JsonElement>(response, Json, x => String(x, "type") is "transcript.text.done" or "error", ct))
        {
            if (request.IncludeRawChunks == true) yield return new TranscriptionRawPart { RawValue = part };
            switch (String(part, "type"))
            {
                case "transcript.text.delta": yield return new TranscriptionDeltaPart { Delta = String(part, "delta") ?? throw new JsonException() }; break;
                case "transcript.text.done":
                    yield return new TranscriptionFinishPart { Text = String(part, "text") ?? throw new JsonException() };
                    yield break;
                case "error": throw new IOException("OpenAI transcription stream failed.");
            }
        }
        throw new IOException("OpenAI transcription stream ended without completion.");
    }

    private static MultipartFormDataContent TranscriptionForm(string model)
    {
        var form = new MultipartFormDataContent();
        form.Add(new StringContent(model), "model");
        return form;
    }

    public Task<JsonElement> CreateClientSecretAsync(JsonObject request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PostAsync<JsonObject, JsonElement>("v1/realtime/client_secrets", request, headers, Json, ct);

    public Task<Streaik.OpenAI.Model.Providers.OpenAI.Live.LiveSessionResponse> CreateLiveSessionAsync(
        Streaik.OpenAI.Model.Providers.OpenAI.Live.LiveSessionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PostAsync<Streaik.OpenAI.Model.Providers.OpenAI.Live.LiveSessionRequest, Streaik.OpenAI.Model.Providers.OpenAI.Live.LiveSessionResponse>(
            "v1/live/sessions", request, headers, Json, ct);

    public async Task<RealtimeTokenResponse> CreateRealtimeTokenAsync(string model, RealtimeTokenRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var session = Session(model, request.SessionConfig);
        var result = await CreateClientSecretAsync(new() { ["session"] = session }, headers, ct);
        return new() { Token = String(result, "value") ?? throw new JsonException("Missing realtime client secret."),
            ExpiresAt = Number(result, "expires_at") is { } expires ? (long)expires : null, Url = $"wss://api.openai.com/v1/realtime?model={Uri.EscapeDataString(model)}" };
    }

    public async Task<RealtimeSessionResponse> CreateRealtimeSessionAsync(string model, RealtimeSessionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var key = RequireKey();
        using var client = clients.CreateClient(Id);
        using var form = new MultipartFormDataContent();
        form.Add(new StringContent(request.Sdp), "sdp");
        form.Add(new StringContent(Session(model, request.SessionConfig).ToJsonString()), "session");
        using var result = await SendAsync(client, HttpMethod.Post, "v1/realtime/calls", form, "Bearer", key, headers, ct);
        var id = result.Headers.Location?.OriginalString.Split('/').LastOrDefault();
        if (string.IsNullOrWhiteSpace(id)) throw new HttpRequestException("Missing OpenAI realtime call ID.");
        return new() { SessionId = id, Sdp = await result.Content.ReadAsStringAsync(ct) };
    }

    public async IAsyncEnumerable<ContainerFileEntry> ListContainerFilesAsync(string containerId, IReadOnlyDictionary<string, string> headers,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        var key = RequireKey();
        using var client = clients.CreateClient(Id);
        var path = $"v1/containers/{Uri.EscapeDataString(containerId)}/files?limit=100&order=desc";
        var cursors = new HashSet<string>(StringComparer.Ordinal);
        string? after = null;
        for (var page = 0; page < 5; page++)
        {
            var uri = after is null ? path : $"{path}&after={Uri.EscapeDataString(after)}";
            var result = await JsonAsync<JsonElement>(client, HttpMethod.Get, uri, null, "Bearer", key, headers, ct);
            if (Property(result, "data") is not { ValueKind: JsonValueKind.Array } files || files.GetArrayLength() == 0) yield break;
            foreach (var file in files.EnumerateArray())
            {
                ct.ThrowIfCancellationRequested();
                if (String(file, "id") is not { Length: > 0 } id) continue;
                long? createdAt = Property(file, "created_at") is { ValueKind: JsonValueKind.Number } stamp && stamp.TryGetInt64(out var seconds)
                    ? seconds : null;
                yield return new(id, String(file, "path"), String(file, "source"), createdAt);
            }
            after = String(result, "last_id");
            if (Property(result, "has_more") is not { ValueKind: JsonValueKind.True } || string.IsNullOrEmpty(after) || !cursors.Add(after)) yield break;
        }
    }

    public async Task<HttpResponseMessage> DownloadContainerFileAsync(string containerId, string fileId, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var key = RequireKey();
        using var client = clients.CreateClient(Id);
        return await SendAsync(client, HttpMethod.Get, $"v1/containers/{Uri.EscapeDataString(containerId)}/files/{Uri.EscapeDataString(fileId)}/content",
            null, "Bearer", key, headers, ct);
    }

    public async Task<bool> IsContainerActiveAsync(string containerId, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var key = RequireKey();
        using var client = clients.CreateClient(Id);
        try
        {
            var result = await JsonAsync<JsonElement>(client, HttpMethod.Get, $"v1/containers/{Uri.EscapeDataString(containerId)}", null, "Bearer", key, headers, ct);
            return String(result, "status") is not ("expired" or "deleted");
        }
        catch (HttpRequestException e) when (e.StatusCode is HttpStatusCode.NotFound or HttpStatusCode.Gone) { return false; }
    }

    private static double? Number(JsonElement root, string name) => Property(root, name) is { ValueKind: JsonValueKind.Number } n ? n.GetDouble() : null;

    private static JsonObject Session(string model, RealtimeSessionConfig? config)
    {
        var session = new JsonObject();
        if (config?.ProviderOptions?.TryGetValue(Id, out var options) == true && options.ValueKind == JsonValueKind.Object)
            session = JsonObject.Create(options)!;
        session["type"] = "realtime";
        session["model"] = model;
        if (config is null) return session;
        if (config.Instructions is { } instructions) session["instructions"] = instructions;
        if (config.OutputModalities is { } modalities) session["output_modalities"] = JsonSerializer.SerializeToNode(modalities, Json);
        if (config.Tools is { } tools) session["tools"] = JsonSerializer.SerializeToNode(tools, Json);
        if (config.Voice is not null || config.InputAudioFormat is not null || config.OutputAudioFormat is not null
            || config.HasTurnDetection || config.InputAudioTranscription is not null || config.OutputAudioTranscription is not null)
        {
            if (config.OutputAudioTranscription is not null) throw new ProviderRequestException("OpenAI does not accept output audio transcription in realtime session configuration.", 422);
            var audio = session["audio"] as JsonObject;
            if (audio is null) { audio = new JsonObject(); session["audio"] = audio; }
            var input = audio["input"] as JsonObject;
            if (input is null) { input = new JsonObject(); audio["input"] = input; }
            var output = audio["output"] as JsonObject;
            if (output is null) { output = new JsonObject(); audio["output"] = output; }
            if (config.Voice is not null) output["voice"] = config.Voice;
            if (config.InputAudioFormat is not null) input["format"] = JsonSerializer.SerializeToNode(config.InputAudioFormat, Json);
            if (config.OutputAudioFormat is not null) output["format"] = JsonSerializer.SerializeToNode(config.OutputAudioFormat, Json);
            if (config.InputAudioTranscription is not null) input["transcription"] = JsonSerializer.SerializeToNode(config.InputAudioTranscription, Json);
            if (config.HasTurnDetection)
            {
                var turn = config.TurnDetection;
                input["turn_detection"] = turn is null ? null : JsonSerializer.SerializeToNode(new { type = turn.Type,
                    threshold = turn.Threshold, silence_duration_ms = turn.SilenceDurationMs, prefix_padding_ms = turn.PrefixPaddingMs }, Json);
            }
        }
        return session;
    }
}
