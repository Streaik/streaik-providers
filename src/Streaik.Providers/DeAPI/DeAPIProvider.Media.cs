using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class DeAPIProvider
{
    private const int ImageLimit = 24 * 1024 * 1024;
    private const int AudioLimit = 32 * 1024 * 1024;
    private const int VideoLimit = 48 * 1024 * 1024;

    public async Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var payload = ImagePayload(request.Model, request.Prompt, request.Size, request.AdditionalProperties);
        using var content = JsonBody(payload);
        var job = await RunJobAsync("images/generations", content, headers, ct);
        return await ImageResultAsync(job, request.N, ct);
    }

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Mask is not null || request.MaskFile is not null || request.Markup is not null)
            throw new ProviderRequestException("DeAPI image edits do not accept masks or markup.", 422);
        var count = (request.ImageFiles?.Length ?? 0) + (request.Images?.Length ?? 0);
        if (count == 0) throw new ProviderRequestException("A DeAPI image edit requires at least one image.");
        var field = count == 1 ? "image" : "images[]";
        using var form = new MultipartFormDataContent();
        foreach (var file in request.ImageFiles ?? []) AddFile(form, field, file);
        foreach (var image in request.Images ?? []) AddDataImage(form, field, image.ImageUrl);
        FormFields(form, ImagePayload(request.Model, request.Prompt, request.Size, request.AdditionalProperties));
        var job = await RunJobAsync("images/edits", form, headers, ct);
        return await ImageResultAsync(job, request.N, ct);
    }

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        yield return NativeMediaOptions.CompletedImage(await GenerateImageAsync(request, headers, ct), edit: false);
    }

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        yield return NativeMediaOptions.CompletedImage(await EditImageAsync(request, headers, ct), edit: true);
    }

    private static Dictionary<string, object?> ImagePayload(string model, string prompt, string? size, IReadOnlyDictionary<string, JsonElement>? options)
    {
        var payload = Options(options, "model", "prompt", "size", "n", "stream", "partial_images", "image", "images", "images[]", "mask", "markup", "response_format");
        var aspectRatio = TakeText(payload, "aspect_ratio") ?? TakeText(payload, "aspectRatio");
        payload.Remove("aspect_ratio");
        payload.Remove("aspectRatio");
        payload["model"] = model;
        if (!string.IsNullOrWhiteSpace(prompt)) payload["prompt"] = prompt;
        AddSize(payload, size, aspectRatio, fallback: 1024);
        return payload;
    }

    private async Task<OpenAIImagesResponse> ImageResultAsync(JsonElement job, int? n, CancellationToken ct)
    {
        var (bytes, mime) = await DownloadAsync(ResultUrl(job), "image", "image/jpeg", ImageLimit, ct);
        var extensions = new Dictionary<string, JsonElement> { ["providerMetadata"] = JsonSerializer.SerializeToElement(Metadata(job), Json) };
        if (n is > 1) extensions["warnings"] = JsonSerializer.SerializeToElement(new[] { new { type = "unsupported", feature = "n" } }, Json);
        return new() { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Data = [new() { B64Json = Convert.ToBase64String(bytes) }],
            OutputFormat = mime == "image/jpeg" ? "jpeg" : mime[6..], AdditionalProperties = extensions };
    }

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var payload = Options(request.AdditionalProperties, "model", "text", "input", "response_format", "stream", "stream_format", "ref_audio");
        var language = TakeText(payload, "language");
        payload.Remove("language");
        payload["model"] = request.Model;
        payload["text"] = request.Input;
        if (request.Voice is { Length: > 0 } voice) payload["voice"] = voice;
        if (language is not null) payload.TryAdd("lang", language);
        if (request.Speed is { } speed) payload["speed"] = speed;
        if (request.ResponseFormat is { Length: > 0 } format) payload["format"] = format;
        var audioFormat = TakeText(payload, "format");
        using var form = new MultipartFormDataContent();
        FormFields(form, payload);
        var job = await RunJobAsync("audio/speech", form, headers, ct);
        var fallback = audioFormat?.ToLowerInvariant() switch
        {
            "mp3" => "audio/mpeg", "flac" => "audio/flac", "opus" => "audio/opus", "ogg" => "audio/ogg", "aac" => "audio/aac",
            "pcm" or "pcm16" => "audio/pcm", _ => "audio/wav"
        };
        var (bytes, mime) = await DownloadAsync(ResultUrl(job), "audio", fallback, AudioLimit, ct);
        var content = new ByteArrayContent(bytes);
        content.Headers.ContentType = new(mime);
        return new SpeechProviderResponse(content, Metadata(job));
    }

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var payload = Options(options, "model", "file", "source_file", "stream", "response_format", "language");
        payload["model"] = model;
        payload.TryAdd("include_ts", false);
        if (language is not null) payload["lang"] = language;
        using var form = new MultipartFormDataContent();
        if (!payload.ContainsKey("source_url")) AddFile(form, "source_file", audio, mediaType, fileName);
        else if (!Uri.TryCreate(TakeText(payload, "source_url"), UriKind.Absolute, out var source) || source.Scheme is not ("http" or "https")
            || source.Host.Length == 0 || source.UserInfo.Length != 0)
            throw new ProviderRequestException("DeAPI source_url must be a valid HTTP(S) URL.");
        FormFields(form, payload);
        var job = await RunJobAsync("audio/transcriptions", form, headers, ct);
        var result = Property(job, "result");
        return new() { Text = result.ValueKind == JsonValueKind.String ? result.GetString()! : String(result, "text")
                ?? String(result, "transcription") ?? String(job, "text") ?? String(job, "transcription") ?? "",
            Language = String(result, "language") ?? TakeText(payload, "lang"), ProviderMetadata = Metadata(job),
            Response = new() { Timestamp = DateTime.UtcNow, ModelId = $"{Id}/{model}" } };
    }

    public async Task<VideoOperationStartResult> StartVideoAsync(VideoRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Image is not null || request.InputReferences?.Any() == true || request.FrameImages?.Any() == true)
            throw new ProviderRequestException("DeAPI video operations support text input only.", 422);
        // VideoRequest.ProviderOptions is already a provider-keyed envelope.
        var scoped = request.ProviderOptions?.TryGetValue(Id, out var options) == true ? new Dictionary<string, JsonElement> { [Id] = options } : null;
        var payload = Options(scoped, "model", "prompt", "image", "images", "input_references", "frame_images");
        payload["model"] = request.Model;
        payload["prompt"] = request.Prompt;
        if (request.Seed is { } seed) payload["seed"] = seed;
        if (request.Fps is { } fps) payload["fps"] = fps;
        if (request.Duration is { } duration) payload["frames"] = duration;
        AddSize(payload, request.Resolution, request.AspectRatio, fallback: 512);
        using var content = JsonBody(payload);
        var (id, data) = await SubmitAsync("videos/generations", content, headers, ct);
        return new() { Operation = JsonSerializer.Serialize(new VideoJob(id, request.Model), Json), ProviderMetadata = Metadata(data),
            Response = new() { Timestamp = DateTime.UtcNow, ModelId = $"{Id}/{request.Model}" },
            Warnings = request.N is > 1 ? [new { type = "unsupported", feature = "n" }] : [] };
    }

    public async Task<VideoOperationStatusResult> GetVideoAsync(string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var reference = JsonSerializer.Deserialize<VideoJob>(operation, Json) ?? throw new ProviderRequestException("Invalid DeAPI operation.");
        var job = await GetJobAsync(reference.Id, headers, ct);
        VideoOperationStatusResult result;
        if (State(job) is "error" or "failed" or "cancelled" or "canceled") result = new VideoOperationErrorResult { Error = JobError(job) };
        else if (State(job) != "done") result = new VideoOperationPendingResult();
        else if (ResultUrl(job) is not { Length: > 0 } url) result = new VideoOperationErrorResult { Error = "DeAPI completed the video without a result URL." };
        else
        {
            var (bytes, mime) = await DownloadAsync(url, "video", "video/mp4", VideoLimit, ct);
            result = new VideoOperationCompletedResult { Videos = [new() { Type = "base64", MediaType = mime, Data = Convert.ToBase64String(bytes) }] };
        }
        result.ProviderMetadata = Metadata(job);
        result.Response = new() { Timestamp = DateTime.UtcNow, ModelId = $"{Id}/{reference.Model}" };
        return result;
    }

    private sealed record VideoJob(string Id, string Model);

    private static string? TakeText(Dictionary<string, object?> payload, string key)
        => payload.TryGetValue(key, out var value) ? value switch { string text => text, JsonElement { ValueKind: JsonValueKind.String } e => e.GetString(), _ => null } : null;

    private static void AddSize(Dictionary<string, object?> payload, string? size, string? ratio, int fallback)
    {
        if (Pair(size, 'x') is { } dimensions)
        {
            payload["width"] = dimensions.X;
            payload["height"] = dimensions.Y;
        }
        else if (Pair(ratio, ':') is { } aspect)
        {
            var scale = fallback / (double)Math.Min(aspect.X, aspect.Y);
            payload.TryAdd("width", (int)Math.Clamp(Math.Round(aspect.X * scale / 8) * 8, 256, 1536));
            payload.TryAdd("height", (int)Math.Clamp(Math.Round(aspect.Y * scale / 8) * 8, 256, 1536));
        }
        else
        {
            payload.TryAdd("width", fallback);
            payload.TryAdd("height", fallback);
        }
    }

    private static (int X, int Y)? Pair(string? value, char separator)
        => value?.ToLowerInvariant().Split(separator) is [var x, var y]
            && int.TryParse(x, CultureInfo.InvariantCulture, out var width) && width > 0
            && int.TryParse(y, CultureInfo.InvariantCulture, out var height) && height > 0 ? (width, height) : null;
}
