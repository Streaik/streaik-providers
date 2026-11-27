using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class AllTokenProvider
{
    private const string VideoPrefix = "atv1_";
    private static readonly HashSet<string> VideoFields = new(StringComparer.OrdinalIgnoreCase)
        { "model", "prompt", "duration", "ratio", "resolution", "seed", "generate_audio", "fps", "n", "frameImages", "inputReferences", "image" };

    public async Task<VideoOperationStartResult> StartVideoAsync(VideoRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var payload = new JsonObject();
        foreach (var (name, value) in NativeMediaOptions.ReadScoped(Id, request.ProviderOptions, VideoFields)) payload[name] = Copy(value);
        payload["model"] = request.Model;
        payload["prompt"] = request.Prompt;
        if (request.Duration is { } duration) payload["duration"] = duration;
        if (request.AspectRatio is { } ratio) payload["ratio"] = ratio;
        if (request.Resolution is { } resolution) payload["resolution"] = resolution;
        if (request.Seed is { } seed) payload["seed"] = seed;
        if (request.GenerateAudio is { } audio) payload["generate_audio"] = audio;

        var content = new JsonArray();
        if (request.Image is { } image) content.Add(Reference(image, "image", "first_frame"));
        foreach (var frame in request.FrameImages ?? [])
            content.Add(Reference(frame.Image, "image", frame.FrameType?.ToLowerInvariant() is "last" or "last_frame" ? "last_frame" : "first_frame"));
        foreach (var item in request.InputReferences ?? [])
        {
            var type = item.MediaType?.Split('/')[0]?.ToLowerInvariant() switch { "video" => "video", "audio" => "audio", _ => "image" };
            content.Add(Reference(item, type, "reference_" + type));
        }
        if (content.Count > 0) payload["content"] = content;

        using var client = CreateClient();
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, "videos/generations", payload, Bearer(ApiKey()), headers, Json, ct);
        if (String(body, "id") is not { Length: > 0 } id || string.IsNullOrWhiteSpace(id) || id is "." or "..")
            throw new JsonException("AllToken returned no valid video job ID.");
        var warnings = new List<object>();
        if (request.Fps is not null) warnings.Add(new { type = "unsupported", feature = "fps" });
        if (request.N is > 1) warnings.Add(new { type = "unsupported", feature = "n" });
        return new() { Operation = VideoPrefix + Convert.ToBase64String(JsonSerializer.SerializeToUtf8Bytes(new VideoJob(id, request.Model), Json))
                .TrimEnd('=').Replace('+', '-').Replace('/', '_'), Warnings = warnings,
            ProviderMetadata = VideoMetadata(body), Response = JobInfo(body, upstream, request.Model) };
    }

    public async Task<VideoOperationStatusResult> GetVideoAsync(string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var job = DecodeJob(operation);
        using var client = CreateClient();
        var (body, upstream) = await ReadTaskAsync(client, TaskPath("videos/generations/", job.TaskId), headers, ct);
        var metadata = VideoMetadata(body);
        var info = JobInfo(body, upstream, job.Model ?? String(body, "model") ?? Id);
        var status = Status(body);
        if (Failed(status)) return new VideoOperationErrorResult { Error = $"AllToken video job {status}.", ProviderMetadata = metadata, Response = info };
        if (status != "completed") return new VideoOperationPendingResult { ProviderMetadata = metadata, Response = info, Warnings = [] };
        if (String(body, "video_url") is not { Length: > 0 } url)
            return new VideoOperationErrorResult { Error = "AllToken completed without video output.", ProviderMetadata = metadata, Response = info };
        var uri = PublicMediaDelivery.RequirePublicHttpsUri(url);
        var fallback = uri.AbsolutePath.EndsWith(".webm", StringComparison.OrdinalIgnoreCase) ? "video/webm" : "video/mp4";
        using var assets = CreateClient(AssetClient);
        var (data, mime) = await PublicMediaDelivery.DownloadAsync(assets, url, fallback, VideoLimit, ct);
        return new VideoOperationCompletedResult { Videos = [new() { Type = "base64", Data = data, MediaType = mime }], ProviderMetadata = metadata, Response = info };
    }

    private static JsonObject Reference(VideoFile file, string type, string role)
    {
        var value = file.GetDataOrUrl();
        var url = file is VideoFileUrl || value.StartsWith("data:", StringComparison.OrdinalIgnoreCase) ? value : $"data:{file.MediaType};base64,{value}";
        var field = type + "_url";
        return new() { ["type"] = field, [field] = new JsonObject { ["url"] = url }, ["role"] = role };
    }

    private static Dictionary<string, JsonElement> VideoMetadata(JsonElement body)
    {
        var metadata = ToObject(body);
        metadata.Remove("video_url"); // Presigned asset URLs are not diagnostic metadata.
        return NativeMediaOptions.Metadata(Id, metadata);
    }

    private static HeaderResponseData JobInfo(JsonElement body, Dictionary<string, string?> headers, string model)
        => new() { Timestamp = Timestamp(body).UtcDateTime, ModelId = model,
            Headers = NativeHttp.SafeResponseMetadataHeaders(headers).ToDictionary(pair => pair.Key, pair => pair.Value ?? "") };

    private static DateTimeOffset Timestamp(JsonElement body)
        => DateTimeOffset.TryParse(String(body, "created_at") ?? String(body, "completed_at"), out var timestamp) ? timestamp : DateTimeOffset.UtcNow;

    private static VideoJob DecodeJob(string operation)
    {
        if (!operation.StartsWith(VideoPrefix, StringComparison.Ordinal)) return new(operation, null);
        var encoded = operation[VideoPrefix.Length..].Replace('-', '+').Replace('_', '/');
        try
        {
            var job = JsonSerializer.Deserialize<VideoJob>(Convert.FromBase64String(encoded.PadRight((encoded.Length + 3) / 4 * 4, '=')), Json);
            if (job is not { TaskId.Length: > 0, Model.Length: > 0 }) throw new JsonException();
            return job;
        }
        catch (Exception e) when (e is FormatException or JsonException) { throw new ProviderRequestException("Invalid AllToken video operation."); }
    }

    private sealed record VideoJob(string TaskId, string? Model);
}
