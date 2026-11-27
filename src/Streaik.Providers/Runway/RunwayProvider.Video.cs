using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class RunwayProvider
{
    private const string Upscale = "magnific_video_upscaler_creative";
    private static readonly HashSet<string> Presets = new(StringComparer.OrdinalIgnoreCase)
        { "game-character", "music-superstar", "game-character-man", "cat-character", "influencer", "tennis-coach", "human-resource", "fashion-designer", "cooking-teacher" };

    public async Task<VideoOperationStartResult> StartVideoAsync(VideoRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var model = request.Model;
        var payload = Options(request.ProviderOptions, "model", "promptText", "prompt", "image", "inputReferences", "frameImages", "fps", "n");
        var references = request.InputReferences?.ToArray() ?? [];
        var warnings = new List<object>();
        void Unsupported(string feature, bool when) { if (when) warnings.Add(new { type = "unsupported", feature }); }
        Unsupported("fps", request.Fps is not null);
        Unsupported("n", request.N is > 1);
        Unsupported("frameImages", request.FrameImages?.Any() == true);
        payload["model"] = model;
        string path;
        if (model.Equals(Upscale, StringComparison.OrdinalIgnoreCase))
        {
            path = "video_upscale";
            payload["model"] = Upscale;
            var video = references.FirstOrDefault(r => HasType(r, "video/")) ?? (HasType(request.Image, "video/") ? request.Image : null);
            if (video is not null) payload["videoUri"] = VideoUri(video);
            RequireNativeString(payload, "videoUri", "Runway upscaling requires a video reference.");
            if (payload["resolution"] is null && request.Resolution is { } resolution) payload["resolution"] = resolution;
            Unsupported("prompt", !string.IsNullOrWhiteSpace(request.Prompt));
            Unsupported("duration", request.Duration is not null);
            Unsupported("seed", request.Seed is not null);
            Unsupported("aspectRatio", request.AspectRatio is not null);
        }
        else if (model.Equals("avatar", StringComparison.OrdinalIgnoreCase) || model.StartsWith("avatar/", StringComparison.OrdinalIgnoreCase))
        {
            path = "avatar_videos";
            payload["model"] = "gwm1_avatars";
            var generic = model.Equals("avatar", StringComparison.OrdinalIgnoreCase);
            if (!generic)
            {
                var id = model["avatar/".Length..];
                payload["avatar"] = Presets.Contains(id) ? new JsonObject { ["type"] = "runway-preset", ["presetId"] = id }
                    : Guid.TryParse(id, out _) ? new JsonObject { ["type"] = "custom", ["avatarId"] = id }
                    : throw new ProviderRequestException("Runway avatar shortcuts require a preset name or custom avatar UUID.");
            }
            if (payload["avatar"] is not JsonObject) throw new ProviderRequestException("Runway requires native avatar options for the generic avatar model.");
            if (!generic || payload["speech"] is null)
            {
                var voice = payload["voice"] ?? (payload["speech"] as JsonObject)?["voice"];
                var audio = references.FirstOrDefault(r => HasType(r, "audio/"));
                if (audio is not null) payload["speech"] = new JsonObject { ["type"] = "audio", ["audio"] = VideoUri(audio) };
                else if (!string.IsNullOrWhiteSpace(request.Prompt))
                {
                    var speech = new JsonObject { ["type"] = "text", ["text"] = request.Prompt };
                    if (voice is not null) speech["voice"] = voice.DeepClone();
                    payload["speech"] = speech;
                }
                else throw new ProviderRequestException("Runway avatar video needs speech text or an audio reference.");
            }
            if (payload["speech"] is not JsonObject) throw new ProviderRequestException("Runway speech options must be an object.");
            payload.Remove("voice");
            Unsupported("resolution", request.Resolution is not null);
            Unsupported("aspectRatio", request.AspectRatio is not null);
            Unsupported("duration", request.Duration is not null);
            Unsupported("seed", request.Seed is not null);
            Unsupported("image", request.Image is not null);
        }
        else
        {
            var image = request.Image ?? references.FirstOrDefault(r => HasType(r, "image/") || HasType(r, "video/"));
            if (image is not null)
            {
                if (HasType(image, "video/")) payload["videoUri"] = VideoUri(image);
                else if (HasType(image, "image/")) payload["promptImage"] = VideoUri(image);
                else throw new ProviderRequestException("Runway video references require an image or video media type.", 415);
            }
            path = payload["videoUri"] is not null ? "video_to_video" : payload["promptImage"] is not null ? "image_to_video" : "text_to_video";
            if (!string.IsNullOrWhiteSpace(request.Prompt)) payload["promptText"] = request.Prompt;
            if (path == "text_to_video" && string.IsNullOrWhiteSpace(request.Prompt)) throw new ProviderRequestException("Runway requires a video prompt or media reference.");
            if (payload["ratio"] is null && (request.Resolution ?? request.AspectRatio) is { } ratio) payload["ratio"] = Ratio(ratio);
            if (request.Duration is { } duration) payload["duration"] = duration;
            if (request.GenerateAudio is { } audio) payload["audio"] = audio;
            if (request.Seed is { } seed && path != "text_to_video") payload["seed"] = seed;
            else Unsupported("seed", request.Seed is not null);
        }

        var (body, upstream) = await SubmitAsync(path, payload, headers, ct);
        return new() { Operation = TaskId(body), Warnings = warnings, ProviderMetadata = NativeMediaOptions.Metadata(Id, body),
            Response = new() { ModelId = model, Timestamp = DateTime.UtcNow,
                Headers = NativeHttp.SafeResponseMetadataHeaders(upstream).ToDictionary(p => p.Key, p => p.Value ?? "") } };
    }

    public async Task<VideoOperationStatusResult> GetVideoAsync(string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(operation) || operation is "." or "..") throw new ProviderRequestException("Invalid Runway task ID.");
        var body = await ReadTaskAsync(operation, ct);
        var metadata = NativeMediaOptions.Metadata(Id, body);
        var response = new HeaderResponseData { Timestamp = DateTime.UtcNow, ModelId = String(body, "model") ?? Id };
        if (Failure(body) is { } failure) return new VideoOperationErrorResult { Error = failure, ProviderMetadata = metadata, Response = response };
        var status = String(body, "status") ?? throw new JsonException("Runway returned no task status.");
        if (!status.Equals("SUCCEEDED", StringComparison.OrdinalIgnoreCase)) return new VideoOperationPendingResult { ProviderMetadata = metadata, Response = response };
        if (Outputs(body).FirstOrDefault() is not { } url)
            return new VideoOperationErrorResult { Error = "Runway completed the task without video output.", ProviderMetadata = metadata, Response = response };
        var (data, mime) = await DownloadAsync(url, "video/mp4", InlineLimit, ct);
        return new VideoOperationCompletedResult { ProviderMetadata = metadata, Response = response,
            Videos = [new() { Type = "base64", Data = data, MediaType = mime }] };
    }

    private static void RequireNativeString(JsonObject payload, string property, string error)
    {
        if (payload[property] is not JsonValue value || !value.TryGetValue<string>(out var text) || string.IsNullOrWhiteSpace(text))
            throw new ProviderRequestException(error);
    }

    private static bool HasType(VideoFile? file, string prefix) => file?.MediaType?.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) == true;
    private static string VideoUri(VideoFile file)
    {
        var source = file is VideoFileUrl url ? url.Url : file.Data;
        if (string.IsNullOrWhiteSpace(source)) throw new ProviderRequestException("Missing Runway media reference.");
        if (source.Length > InlineLimit * 4 / 3 + 128) throw new ProviderRequestException("Runway input exceeds the inline size limit.", 413);
        return MediaUri(source, file.MediaType ?? "application/octet-stream");
    }
}
