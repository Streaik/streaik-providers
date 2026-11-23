using System.Text.Json;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.BlackForestLabs;

internal sealed partial class BlackForestLabsProvider
{
    private sealed record VideoOperation(string Model, string PollingUrl);

    public async Task<VideoOperationStartResult> StartVideoAsync(VideoRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var model = request.Model.Trim().ToLowerInvariant();
        if (!VideoModels.Contains(model)) throw new ProviderRequestException("Unsupported Black Forest Labs video model.");
        var (endpoint, payload, warnings) = VideoPayload(request, model);
        using var client = clients.CreateClient(Id);
        var (body, responseHeaders) = await NativeHttp.PostJsonAsync(client, endpoint, payload, Auth(), headers, Json, ct);
        var (_, target) = PollTarget(body);
        return new()
        {
            // The gateway wraps this value in a protected, model-scoped reference.
            Operation = JsonSerializer.Serialize(new VideoOperation(model, target.AbsoluteUri), Json),
            Warnings = warnings,
            ProviderMetadata = Metadata(body),
            Response = new() { ModelId = model, Timestamp = DateTime.UtcNow,
                Headers = responseHeaders.Where(pair => pair.Value is not null).ToDictionary(pair => pair.Key, pair => pair.Value!, StringComparer.OrdinalIgnoreCase) }
        };
    }

    public async Task<VideoOperationStatusResult> GetVideoAsync(string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        VideoOperation reference;
        try { reference = JsonSerializer.Deserialize<VideoOperation>(operation, Json) ?? throw new JsonException(); }
        catch (JsonException) { throw new ProviderRequestException("Invalid Black Forest Labs video operation."); }
        if (!VideoModels.Contains(reference.Model) || string.IsNullOrEmpty(reference.PollingUrl))
            throw new ProviderRequestException("Invalid Black Forest Labs video operation.");
        using var client = clients.CreateClient(Id);
        var (status, body) = await PollAsync(client, PollUri(reference.PollingUrl), Auth(), ct);
        var response = new HeaderResponseData { ModelId = reference.Model, Timestamp = DateTime.UtcNow };
        var terminal = Ready(status) || Failed(status);
        var metadata = Metadata(WithoutMedia(body, ["sample"]), terminal ? Cost(body) : null);
        if (Failed(status)) return new VideoOperationErrorResult { Error = $"Black Forest Labs video task failed: {status}.", Response = response, ProviderMetadata = metadata };
        if (!terminal) return new VideoOperationPendingResult { Response = response, ProviderMetadata = metadata };

        var sample = String(Property(body, "result"), "sample");
        if (string.IsNullOrWhiteSpace(sample))
            return new VideoOperationErrorResult { Error = "Black Forest Labs did not return a video.", Response = response, ProviderMetadata = metadata };
        var (data, mime) = !sample.StartsWith("data:", StringComparison.OrdinalIgnoreCase) && Uri.TryCreate(sample, UriKind.Absolute, out var url)
            ? await DownloadAsync(url, Path.GetExtension(url.AbsolutePath).ToLowerInvariant() switch
                { ".webm" => "video/webm", ".mov" => "video/quicktime", _ => "video/mp4" }, VideoLimit, ct)
            : Inline(sample, "video/mp4", VideoLimit);
        return new VideoOperationCompletedResult { Videos = [new() { Type = "base64", MediaType = mime, Data = data }], Response = response, ProviderMetadata = metadata };
    }

    private static (string Endpoint, Dictionary<string, object?> Payload, List<object> Warnings) VideoPayload(VideoRequest request, string model)
    {
        var tool = model.StartsWith("flux-tools/", StringComparison.Ordinal);
        var draft = model == "flux-3-video-draft-enhance";
        var references = request.InputReferences?.ToArray() ?? [];
        var frames = request.FrameImages?.ToArray() ?? [];
        var imageInput = request.Image is not null || frames.Length > 0;
        var mode = draft ? "draft_enhance" : references.Length > 0 ? "v2v" : imageInput ? "i2v" : "t2v";
        var payload = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
        var warnings = new List<object>();
        if (request.Fps is not null) warnings.Add(Warning("fps"));
        if (request.N is > 1) warnings.Add(Warning("n"));
        if (request.Seed is not null) warnings.Add(Warning("seed"));
        if (references.Length > 1) warnings.Add(Warning("inputReferences", "Only the first video reference is used."));
        if (draft || tool)
        {
            if (request.AspectRatio is not null) warnings.Add(Warning("aspect_ratio"));
            if (request.Duration is not null) warnings.Add(Warning("duration"));
            if (request.GenerateAudio is not null) warnings.Add(Warning("generate_audio"));
            if (imageInput) warnings.Add(Warning("image inputs"));
        }
        if (request.ProviderOptions?.TryGetValue(Id, out var options) == true && options.ValueKind != JsonValueKind.Null)
        {
            if (options.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Black Forest Labs provider options must be an object.");
            foreach (var property in options.EnumerateObject())
                if (property.Name.ToLowerInvariant() is not ("model" or "mode" or "prompt" or "stream" or "n" or "image" or "inputreferences" or "frameimages"))
                    payload[property.Name] = property.Value;
        }

        if (tool)
        {
            if (request.Resolution is not null) warnings.Add(Warning("resolution"));
            var edit = model == "flux-tools/video-edit-v1";
            var key = edit ? "video" : "input_video";
            if (references.FirstOrDefault() is { } source) Set(key, VideoInput(source));
            if (!HasNativeValue(payload, key)) throw new ProviderRequestException($"'{key}' is required for this FLUX tool.");
            if (edit && (string.IsNullOrWhiteSpace(request.Prompt) || request.Prompt.Length > 4096))
                throw new ProviderRequestException("Video editing requires a prompt of 1 to 4096 characters.");
            if (!string.IsNullOrWhiteSpace(request.Prompt)) Set("prompt", request.Prompt);
            return ($"v1/{model}", payload, warnings);
        }

        Set("mode", mode);
        if (!string.IsNullOrWhiteSpace(request.Resolution)) Set("resolution", request.Resolution);
        if (draft)
        {
            if (!string.IsNullOrWhiteSpace(request.Prompt)) warnings.Add(Warning("prompt"));
            if (references.FirstOrDefault() is { } source) Set("draft_cache", VideoInput(source));
            if (!HasNativeValue(payload, "draft_cache")) throw new ProviderRequestException("Draft enhance requires a draft cache.");
        }
        else
        {
            if (string.IsNullOrWhiteSpace(request.Prompt)) throw new ProviderRequestException("'prompt' is required.");
            Set("prompt", request.Prompt);
            if (!string.IsNullOrWhiteSpace(request.AspectRatio)) Set("aspect_ratio", request.AspectRatio);
            if (request.Duration is { } duration) Set("duration", duration);
            if (request.GenerateAudio is { } audio) Set("generate_audio", audio);
            if (mode == "v2v") Set("start_video", VideoInput(references[0]));
            else if (mode == "i2v") Set("keyframes", Keyframes(request.Image, frames));
        }
        return ("v1/flux-3-video", payload, warnings);

        void Set(string key, object value) { payload.Remove(key); payload[key] = value; }
    }

    private static bool HasNativeValue(Dictionary<string, object?> payload, string key) => payload.GetValueOrDefault(key) switch
    {
        string text => !string.IsNullOrWhiteSpace(text),
        JsonElement { ValueKind: JsonValueKind.String } element => !string.IsNullOrWhiteSpace(element.GetString()),
        JsonElement { ValueKind: not (JsonValueKind.Null or JsonValueKind.Undefined) } => true,
        _ => false
    };

    private static string[] Keyframes(VideoFile? first, VideoFrameImage[] frames)
    {
        if (frames.Length > 10) throw new ProviderRequestException("FLUX 3 video supports at most 10 keyframes.");
        VideoFile? last = null;
        var ordered = new List<VideoFile>();
        foreach (var frame in frames)
        {
            var image = frame?.Image ?? throw new ProviderRequestException("Every keyframe must include an image.");
            switch (frame!.FrameType?.ToLowerInvariant())
            {
                case "first" or "first_frame" or "firstframe": first = image; break;
                case "last" or "last_frame" or "lastframe": last = image; break;
                default: ordered.Add(image); break;
            }
        }
        if (first is not null) ordered.Insert(0, first);
        if (last is not null) ordered.Add(last);
        if (ordered.Count is < 1 or > 10) throw new ProviderRequestException("FLUX 3 video requires between 1 and 10 keyframes.");
        return ordered.Select(VideoInput).ToArray();
    }

    private static string VideoInput(VideoFile input) => Reference(input is VideoFileUrl url ? url.Url : input.Data);
}
