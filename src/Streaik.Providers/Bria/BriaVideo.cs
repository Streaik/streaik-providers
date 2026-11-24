using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class BriaProvider
{
    private static readonly Dictionary<string, string> VideoModels = new(StringComparer.Ordinal)
    {
        ["video/edit/erase"] = "Video Erase",
        ["video/edit/increase_resolution"] = "Video Increase Resolution",
        ["video/edit/remove_background"] = "Video Remove Background",
        ["video/edit/replace_background"] = "Video Replace Background",
        ["video/edit/green_screen"] = "Video Green Screen",
        ["video/segment/mask_by_prompt"] = "Video Mask by Prompt",
        ["video/segment/mask_by_key_points"] = "Video Mask by Key Points",
        ["video/segment/foreground_mask"] = "Video Foreground Mask"
    };

    public async Task<VideoOperationStartResult> StartVideoAsync(VideoRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var model = request.Model;
        if (!VideoModels.ContainsKey(model)) throw new ProviderRequestException("Unknown Bria video model.");
        var payload = new JsonObject();
        if (request.ProviderOptions?.TryGetValue(Id, out var options) == true && options.ValueKind != JsonValueKind.Null)
        {
            if (options.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Bria video options must be an object.");
            foreach (var p in options.EnumerateObject())
                if (p.Name.ToLowerInvariant() is not ("model" or "stream" or "api_token" or "status_url" or "request_id")) payload[p.Name] = Copy(p.Value);
        }
        var references = request.InputReferences?.ToList() ?? [];
        var candidates = references.Prepend(request.Image).OfType<VideoFile>();
        if (candidates.FirstOrDefault(IsVideo) is { } video)
        {
            payload["video"] = VideoInput(video);
            references.Remove(video);
        }
        if (!HasText(payload, "video")) throw new ProviderRequestException("Bria requires a video input.");
        switch (model)
        {
            case "video/edit/erase":
                if (references.FirstOrDefault(IsVisual) is { } mask)
                {
                    payload["mask"] = new JsonObject { ["mask_url"] = VideoInput(mask) };
                    references.Remove(mask);
                }
                if (payload["mask"] is null) throw new ProviderRequestException("Bria video erase requires a mask.");
                break;
            case "video/edit/replace_background":
                if (references.FirstOrDefault(IsVisual) is { } background)
                {
                    payload["background_url"] = VideoInput(background);
                    references.Remove(background);
                }
                if (!HasText(payload, "background_url")) throw new ProviderRequestException("Bria video background replacement requires a background input.");
                break;
            case "video/segment/mask_by_prompt":
                if (!string.IsNullOrWhiteSpace(request.Prompt)) payload["prompt"] = request.Prompt;
                if (!HasText(payload, "prompt")) throw new ProviderRequestException("Bria mask by prompt requires a prompt.");
                break;
            case "video/segment/mask_by_key_points":
                if (payload["frame_index"] is null || payload["key_points"] is not JsonArray { Count: > 0 })
                    throw new ProviderRequestException("Bria mask by key points requires frame_index and key_points.");
                break;
        }
        var warnings = new List<object>();
        void Warn(bool unsupported, string feature) { if (unsupported) warnings.Add(new { type = "unsupported", feature }); }
        Warn(!string.IsNullOrWhiteSpace(request.Prompt) && model != "video/segment/mask_by_prompt", "prompt");
        Warn(request.Seed is not null, "seed");
        Warn(request.Duration is not null, "duration");
        Warn(request.Fps is not null, "fps");
        Warn(request.Resolution is not null, "resolution");
        Warn(request.AspectRatio is not null, "aspectRatio");
        Warn(request.N is > 1, "n");
        Warn(request.FrameImages?.Any() == true, "frameImages");
        Warn(request.Image is { } image && !IsVideo(image), "image.nonVideo");
        Warn(references.Count > 0, "inputReferences.unused");
        if (request.GenerateAudio is { } audio)
        {
            if (model is "video/edit/erase" or "video/edit/replace_background" or "video/edit/green_screen" or "video/edit/increase_resolution")
                payload["preserve_audio"] = audio;
            else Warn(true, "generateAudio");
        }

        using var client = clients.CreateClient(Id);
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, model, payload, Auth(), headers, Json, ct);
        if (IsError(body)) throw new IOException(Error(body));
        var target = PollTarget(String(body, "request_id"), String(body, "status_url"));
        return new() { Operation = target.PathAndQuery, Warnings = warnings, ProviderMetadata = Scoped(body), Response = Response(model, upstream) };
    }

    public async Task<VideoOperationStatusResult> GetVideoAsync(string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var target = PollTarget(null, operation);
        using var client = clients.CreateClient(Id);
        var (body, upstream) = await GetJobAsync(client, target, Auth(), headers, ct);
        var metadata = Response("", upstream);
        var scoped = Scoped(body);
        if (IsError(body)) return new VideoOperationErrorResult { Error = Error(body), Response = metadata, ProviderMetadata = scoped };
        if (Status(body) != "COMPLETED") return new VideoOperationPendingResult { Response = metadata, ProviderMetadata = scoped };
        var result = Property(body, "result");
        var url = VideoOutput(result) ?? VideoOutput(body);
        if (url is null) return new VideoOperationErrorResult { Error = "Bria returned no video or mask URL.", Response = metadata, ProviderMetadata = scoped };
        var (data, mime) = await DownloadAsync(url, "video/", VideoLimit, ct);
        return new VideoOperationCompletedResult
        {
            Videos = [new() { Type = "base64", Data = data, MediaType = mime }], Response = metadata, ProviderMetadata = scoped
        };
    }

    private static HeaderResponseData Response(string model, Dictionary<string, string?> headers) => new()
    {
        ModelId = model, Timestamp = DateTime.UtcNow,
        Headers = headers.Where(x => x.Value is not null).ToDictionary(x => x.Key, x => x.Value!, StringComparer.OrdinalIgnoreCase)
    };

    private static bool IsVideo(VideoFile file) => file.MediaType?.StartsWith("video/", StringComparison.OrdinalIgnoreCase) == true
        || file is VideoFileUrl url && FileExtension(url.Url) is ".mp4" or ".webm" or ".mov" or ".avi" or ".mkv";
    private static bool IsVisual(VideoFile file) => IsVideo(file) || file.MediaType?.StartsWith("image/", StringComparison.OrdinalIgnoreCase) == true
        || file is VideoFileUrl;
    private static string FileExtension(string? url) => Uri.TryCreate(url, UriKind.Absolute, out var uri) ? Path.GetExtension(uri.AbsolutePath).ToLowerInvariant() : "";

    private static string VideoInput(VideoFile file)
    {
        if (file is VideoFileUrl url && Uri.TryCreate(url.Url, UriKind.Absolute, out var target) && target.Scheme is "http" or "https") return url.Url;
        if (string.IsNullOrWhiteSpace(file.Data) || string.IsNullOrWhiteSpace(file.MediaType)) throw new ProviderRequestException("Invalid Bria video reference.");
        return file.Data.StartsWith("data:", StringComparison.OrdinalIgnoreCase) ? file.Data : $"data:{file.MediaType};base64,{file.Data}";
    }

    private static string? VideoOutput(JsonElement body)
    {
        if (body.ValueKind == JsonValueKind.Object)
        {
            foreach (var name in new[] { "video_url", "mask_url", "output_url", "url" })
                if (String(body, name) is { Length: > 0 } url) return url;
            foreach (var value in body.EnumerateObject())
                if (VideoOutput(value.Value) is { } url) return url;
        }
        else if (body.ValueKind == JsonValueKind.Array)
            foreach (var value in body.EnumerateArray())
                if (VideoOutput(value) is { } url) return url;
        return null;
    }
}
