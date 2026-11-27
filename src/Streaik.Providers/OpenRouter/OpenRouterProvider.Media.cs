using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.OpenRouter;

public sealed partial class OpenRouterProvider
{
    private const int MaxInlineMediaBytes = 32 * 1024 * 1024;

    public async Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => ImageProviderUsage.Read(await PostAsync<JsonObject, JsonElement>("v1/images", ImageBody(ToObject(request)), headers, ct), Json);

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => ImageProviderUsage.Read(await PostAsync<JsonObject, JsonElement>("v1/images", await EditBodyAsync(request, ct), headers, ct), Json);

    public IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamAsync<JsonObject, JsonElement>("v1/images", ImageBody(ToObject(request)), headers, Json, _ => false, ct);

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var body = await EditBodyAsync(request, ct);
        await foreach (var part in StreamAsync<JsonObject, JsonElement>("v1/images", body, headers, Json, _ => false, ct))
        {
            if (String(part, "type") is { } type && type.StartsWith("image_generation.", StringComparison.Ordinal))
            {
                var edit = (JsonObject)Copy(part)!;
                edit["type"] = "image_edit." + type["image_generation.".Length..];
                yield return JsonSerializer.SerializeToElement(edit, Json);
            }
            else yield return part;
        }
    }

    private static JsonObject ImageBody(JsonObject body)
    {
        body.Remove("partial_images"); // The OpenRouter route does not expose this OpenAI-specific knob.
        return body;
    }

    private static async Task<JsonObject> EditBodyAsync(OpenAIImageEditRequest request, CancellationToken ct)
    {
        var body = ImageBody(ToObject(request));
        body.Remove("images"); body.Remove("mask");
        var references = new JsonArray();
        foreach (var image in request.Images ?? []) references.Add(ImageReference(image.ImageUrl));
        foreach (var file in request.ImageFiles ?? []) references.Add(ImageReference(await ImageDataAsync(file, ct)));
        if (references.Count == 0) throw new ProviderRequestException("At least one image is required.");
        body["input_references"] = references;
        if (request.MaskFile is { } mask) body["mask"] = ImageReference(await ImageDataAsync(mask, ct));
        else if (request.Mask is not null) body["mask"] = ImageReference(request.Mask.ImageUrl);
        return body;
    }

    private static JsonObject ImageReference(string? url)
    {
        if (url is null || !(url.StartsWith("data:image/", StringComparison.Ordinal) || Uri.TryCreate(url, UriKind.Absolute, out var uri)
            && uri.Scheme == "https" && uri.UserInfo.Length == 0))
            throw new ProviderRequestException("OpenRouter requires image data URLs or HTTPS image URLs.");
        return new() { ["type"] = "image_url", ["image_url"] = new JsonObject { ["url"] = url } };
    }

    private static async Task<string> ImageDataAsync(IFormFile file, CancellationToken ct)
    {
        if (!file.ContentType.StartsWith("image/", StringComparison.OrdinalIgnoreCase)) throw new ProviderRequestException("An image file is required.");
        await using var input = file.OpenReadStream();
        return $"data:{file.ContentType};base64,{await BoundedBase64Async(input, MaxInlineMediaBytes, ct)}";
    }

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        return await MediaHttp.SendAsync(client, HttpMethod.Post, "v1/audio/speech", JsonContent.Create(request, options: Json),
            "Bearer", RequiredKey(credentials, Id), headers, ct);
    }

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var format = mediaType.Split(';')[0].Trim().ToLowerInvariant() switch
        {
            "audio/mpeg" or "audio/mp3" => "mp3", "audio/wav" or "audio/x-wav" or "audio/wave" => "wav",
            "audio/mp4" or "audio/m4a" or "audio/x-m4a" => "m4a", "audio/flac" or "audio/x-flac" => "flac",
            "audio/ogg" or "audio/opus" => "ogg", "audio/webm" => "webm", "audio/aac" => "aac",
            _ => throw new ProviderRequestException("Use containerized audio (mp3, wav, m4a, flac, ogg, webm, or aac).", 422)
        };
        var body = new JsonObject();
        if (options is not null)
            foreach (var (name, value) in options)
                if (name is not ("model" or "file" or "input_audio" or "stream" or "response_format") && value.ValueKind != JsonValueKind.Undefined)
                    body[name] = Copy(value);
        body["model"] = model;
        if (!string.IsNullOrWhiteSpace(language)) body["language"] = language;
        body["input_audio"] = new JsonObject { ["format"] = format, ["data"] = await BoundedBase64Async(audio, MaxInlineMediaBytes, ct) };
        var result = await PostAsync<JsonObject, JsonElement>("v1/audio/transcriptions", body, headers, ct);
        return new TranscriptionResponse
        {
            Text = String(result, "text") ?? throw new JsonException("Missing OpenRouter transcript."),
            Language = String(result, "language"), DurationInSeconds = MediaNumber(result, "duration"),
            Segments = Array(Property(result, "segments")).Where(x => String(x, "text") is not null).Select(x => new TranscriptionSegment
                { Text = String(x, "text")!, StartSecond = MediaNumber(x, "start") ?? 0, EndSecond = MediaNumber(x, "end") ?? 0 }).ToArray(),
            ProviderMetadata = Property(result, "usage").ValueKind != JsonValueKind.Undefined ? new() { [Id] = Property(result, "usage") } : null,
            Response = new() { Timestamp = DateTime.UtcNow }
        };
    }

    public async Task<VideoOperationStartResult> StartVideoAsync(VideoRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var body = new JsonObject();
        if (request.ProviderOptions?.TryGetValue(Id, out var options) == true && options.ValueKind == JsonValueKind.Object)
            foreach (var item in options.EnumerateObject()) body[item.Name] = Copy(item.Value);
        body["model"] = request.Model;
        body["prompt"] = request.Prompt;
        if (request.AspectRatio is { } ratio) body["aspect_ratio"] = ratio;
        if (request.Duration is { } duration) body["duration"] = duration;
        if (request.Resolution is { } resolution) body["resolution"] = resolution;
        if (request.Seed is { } seed) body["seed"] = seed;
        if (request.GenerateAudio is { } audio) body["generate_audio"] = audio;
        if (request.Fps is not null || request.N is > 1) throw new ProviderRequestException("OpenRouter video does not support explicit fps or multiple generations per job.", 422);
        var references = request.InputReferences?.ToArray() ?? [];
        if (references.Length == 0 && request.Image is { } image) references = [image];
        if (references.Length > 0) body["input_references"] = new JsonArray(references.Select(VideoReference).ToArray());
        if (request.FrameImages?.ToArray() is { Length: > 0 } frames)
            body["frame_images"] = new JsonArray(frames.Select(frame =>
            {
                var reference = VideoReference(frame.Image);
                reference["frame_type"] = frame.FrameType switch { "first_frame" or "firstFrame" or "first" => "first_frame",
                    "last_frame" or "lastFrame" or "last" => "last_frame", _ => throw new ProviderRequestException("Invalid video frame type.") };
                return reference;
            }).ToArray());
        var result = await PostAsync<JsonObject, JsonElement>("v1/videos", body, headers, ct);
        var id = String(result, "id");
        if (!ValidVideoId(id)) throw new JsonException("Invalid OpenRouter video ID.");
        return new() { Operation = id!, Response = new() { ModelId = request.Model, Timestamp = DateTime.UtcNow } };
    }

    public async Task<VideoOperationStatusResult> GetVideoAsync(string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!ValidVideoId(operation)) throw new ProviderRequestException("Invalid OpenRouter video operation.");
        var path = $"v1/videos/{operation}";
        using var client = clients.CreateClient(Id);
        var auth = Auth();
        var result = await NativeHttp.GetAsync<JsonElement>(client, path, auth, ct);
        var model = String(result, "model") ?? Id;
        switch (String(result, "status"))
        {
            case "pending": case "queued": case "in_progress": case "processing":
                return new VideoOperationPendingResult { Response = new() { ModelId = model, Timestamp = DateTime.UtcNow } };
            case "completed": break;
            default: return new VideoOperationErrorResult { Error = "OpenRouter video generation did not complete successfully.",
                Response = new() { ModelId = model, Timestamp = DateTime.UtcNow } };
        }
        var count = Math.Max(1, Array(Property(result, "unsigned_urls")).Count());
        if (count > 8) throw new JsonException("Unexpected number of OpenRouter video outputs.");
        var videos = new List<VideoOperationVideoData>(count);
        for (var i = 0; i < count; i++)
        {
            using var message = new HttpRequestMessage(HttpMethod.Get, $"{path}/content?index={i}");
            auth(message.Headers);
            using var response = await NativeHttp.SendAsync(client, message, ct);
            var mime = response.Content.Headers.ContentType?.MediaType;
            if (mime is not (null or "application/octet-stream") && !mime.StartsWith("video/", StringComparison.OrdinalIgnoreCase))
                throw new JsonException("OpenRouter content is not a video.");
            await using var input = await response.Content.ReadAsStreamAsync(ct);
            videos.Add(new() { Type = "base64", MediaType = mime is null or "application/octet-stream" ? "video/mp4" : mime,
                Data = await BoundedBase64Async(input, MaxInlineMediaBytes / count, ct) });
        }
        var usage = Property(result, "usage");
        return new VideoOperationCompletedResult { Videos = videos, Response = new() { ModelId = model, Timestamp = DateTime.UtcNow },
            ProviderMetadata = usage.ValueKind == JsonValueKind.Object
                ? new() { [Id] = JsonSerializer.SerializeToElement(new { usage }, Json) } : null };
    }

    private static bool ValidVideoId(string? value) => value is { Length: > 0 and <= 256 } && value.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_');
    private static JsonObject VideoReference(VideoFile file) => ImageReference(file is VideoFileUrl url ? url.Url
        : file.Data?.StartsWith("data:", StringComparison.Ordinal) == true ? file.Data : $"data:{file.MediaType ?? "image/png"};base64,{file.Data}");
    private static double? MediaNumber(JsonElement value, string name) => Property(value, name) is { ValueKind: JsonValueKind.Number } number
        && number.TryGetDouble(out var d) && double.IsFinite(d) ? d : null;

}
