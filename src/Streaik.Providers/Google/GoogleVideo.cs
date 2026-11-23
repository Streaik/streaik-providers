using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.WebUtilities;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class GoogleGeminiProvider
{
    public async Task<VideoOperationStartResult> StartVideoAsync(VideoRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var model = NormalizeModel(request.Model);
        var omni = model.Contains("-omni-", StringComparison.OrdinalIgnoreCase);
        if (!omni && !model.StartsWith("veo-", StringComparison.OrdinalIgnoreCase))
            throw new ProviderRequestException("Use a Veo or Gemini Omni video model.", 422);
        if (request.Fps is not null || request.Seed is not null || request.N is < 1 or > 4 || omni && request.N is > 1)
            throw new ProviderRequestException("Unsupported video generation options for this Google model.", 422);
        if (request.Duration is { } duration && (duration < 1 || duration > 30 || duration != Math.Truncate(duration)))
            throw new ProviderRequestException("Google video duration must be specified in whole seconds.", 422);
        var frames = request.FrameImages?.ToArray() ?? [];
        if (frames.Any(x => x is null || x.Image is null)) throw new ProviderRequestException("A frame must contain an image.", 422);
        var groups = frames.GroupBy(x => x.FrameType switch
        {
            "first_frame" or "firstFrame" or "first" => "first", "last_frame" or "lastFrame" or "last" => "last",
            _ => throw new ProviderRequestException("Invalid video frame type.", 422)
        }).ToDictionary(g => g.Key, g => g.Count() == 1 ? g.Single().Image : throw new ProviderRequestException("Duplicate video frame type.", 422));
        var first = groups.GetValueOrDefault("first") ?? request.Image;
        var last = groups.GetValueOrDefault("last");
        if (last is not null && first is null) throw new ProviderRequestException("A last frame requires a first frame.", 422);
        var references = request.InputReferences?.ToArray() ?? [];
        JsonElement created;
        string operation;
        if (omni)
        {
            var input = new JsonArray(TextPart(request.Prompt));
            if (first is not null) input.Add(ToGoogleVideoImage(first));
            if (last is not null) input.Add(ToGoogleVideoImage(last));
            foreach (var reference in references) input.Add(ToGoogleVideoImage(reference));
            var format = new JsonObject { ["type"] = "video" };
            if (request.Duration is { } seconds) format["duration"] = $"{(int)seconds}s";
            if (request.AspectRatio is { } ratio) format["aspect_ratio"] = ratio;
            if (request.Resolution is { } resolution) format["resolution"] = resolution;
            if (request.GenerateAudio is { } audio) format["generate_audio"] = audio;
            created = await RunInteractionAsync(new() { ["model"] = model, ["input"] = input, ["response_format"] = format,
                ["background"] = true, ["store"] = true, ["stream"] = false }, ct);
            operation = "omni_" + ResourceId(String(created, "id") ?? throw new JsonException("Missing Google interaction ID."));
        }
        else
        {
            var instance = new JsonObject { ["prompt"] = request.Prompt };
            if (first is not null) instance["image"] = ToVeoInlineImage(first);
            if (last is not null) instance["lastFrame"] = ToVeoInlineImage(last);
            if (references.Length > 0) instance["referenceImages"] = new JsonArray(references.Select(x => new JsonObject
                { ["image"] = ToVeoInlineImage(x), ["referenceType"] = "asset" }).ToArray());
            var parameters = new JsonObject();
            if (request.ProviderOptions?.TryGetValue("google", out var options) == true && options.ValueKind == JsonValueKind.Object)
                foreach (var item in options.EnumerateObject())
                    if (item.Name is "negativePrompt" or "personGeneration") parameters[item.Name] = Copy(item.Value);
            if (request.AspectRatio is { } ratio) parameters["aspectRatio"] = ratio;
            if (request.Resolution is { } resolution) parameters["resolution"] = resolution;
            if (request.Duration is { } seconds) parameters["durationSeconds"] = (int)seconds;
            if (request.N is { } n) parameters["sampleCount"] = n;
            if (request.GenerateAudio is { } audio) parameters["generateAudio"] = audio;
            created = await PostGoogleJsonAsync($"v1beta/models/{model}:predictLongRunning", new()
                { ["instances"] = new JsonArray(instance), ["parameters"] = parameters }, ct);
            var name = String(created, "name") ?? throw new JsonException("Missing Veo operation name.");
            _ = VeoOperationPath(name);
            operation = "veo_" + WebEncoders.Base64UrlEncode(Encoding.UTF8.GetBytes(name));
        }
        return new() { Operation = operation, Response = new() { ModelId = model, Timestamp = DateTime.UtcNow } };
    }

    public async Task<VideoOperationStatusResult> GetVideoAsync(string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        var key = ApiKey();
        JsonElement status;
        string model;
        IEnumerable<JsonElement> parts;
        if (operation.StartsWith("omni_", StringComparison.Ordinal))
        {
            using var response = await GetInteractionAsync(operation[5..], ct);
            status = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
            model = String(status, "model") ?? "google";
            var current = String(status, "status");
            if (current is "in_progress" or "queued" or "pending") return Pending(model);
            if (current != "completed") return Failure(model);
            parts = OutputParts(status).Where(x => String(x, "type") == "video");
        }
        else if (operation.StartsWith("veo_", StringComparison.Ordinal) && operation.Length <= 1024)
        {
            string name;
            try { name = Encoding.UTF8.GetString(WebEncoders.Base64UrlDecode(operation[4..])); }
            catch (FormatException) { throw new ProviderRequestException("Invalid Veo operation."); }
            var path = VeoOperationPath(name);
            model = path.Split('/')[2];
            status = await NativeHttp.GetAsync<JsonElement>(client, path, GoogleAuth(key), ct);
            if (Property(status, "error").ValueKind == JsonValueKind.Object) return Failure(model);
            if (Property(status, "done").ValueKind != JsonValueKind.True) return Pending(model);
            parts = Array(Property(Property(Property(status, "response"), "generateVideoResponse"), "generatedSamples"))
                .Select(sample => Property(sample, "video"));
        }
        else throw new ProviderRequestException("Invalid Google video operation.");

        var data = parts.ToArray();
        if (data.Length is 0 or > 8) return Failure(model);
        var videos = new List<VideoOperationVideoData>();
        foreach (var part in data)
        {
            var mime = String(part, "mime_type") ?? "video/mp4";
            var inline = String(part, "data");
            if (inline is not null && inline.Length > InlineLimit * 4L / 3 + 4) throw new JsonException("Google video output is too large.");
            if (inline is null && String(part, "uri") is { } uri)
                (inline, mime) = await DownloadGoogleVideoAsync(client, key, uri, InlineLimit / data.Length, ct);
            if (inline is not null) videos.Add(new() { Type = "base64", Data = inline, MediaType = mime });
        }
        return videos.Count == 0 ? Failure(model) : new VideoOperationCompletedResult { Videos = videos,
            Response = VideoResponse(model) };
    }

    private static string VeoOperationPath(string name)
    {
        if (name.Split('/') is not ["models", var model, "operations", var id] || !model.StartsWith("veo-", StringComparison.Ordinal))
            throw new ProviderRequestException("Invalid Google Veo operation name.");
        return $"v1beta/models/{ResourceId(model)}/operations/{ResourceId(id)}";
    }

    private static JsonObject ToGoogleVideoImage(VideoFile file) => file is VideoFileUrl url
        ? GoogleImagePart(url.Url, file.MediaType) : GoogleImagePart(file.Data, file.MediaType ?? "image/png");

    private static JsonObject ToVeoInlineImage(VideoFile file)
    {
        var part = ToGoogleVideoImage(file);
        if (part["data"] is null) throw new ProviderRequestException("Veo image inputs must be inline data.", 422);
        return new() { ["bytesBase64Encoded"] = part["data"]!.DeepClone(), ["mimeType"] = part["mime_type"]?.DeepClone() };
    }

    private static HeaderResponseData VideoResponse(string model) => new() { ModelId = model, Timestamp = DateTime.UtcNow };
    private static VideoOperationPendingResult Pending(string model) => new() { Response = VideoResponse(model) };
    private static VideoOperationErrorResult Failure(string model) => new() { Error = "Google video generation did not return a video.", Response = VideoResponse(model) };

    private static async Task<(string Data, string Mime)> DownloadGoogleVideoAsync(HttpClient client, string key, string source, int limit, CancellationToken ct)
    {
        if (!Uri.TryCreate(source, UriKind.Absolute, out var uri) || !AllowedGoogleDownload(uri) || uri.Host != "generativelanguage.googleapis.com")
            throw new JsonException("Invalid Google video download URL.");
        for (var hop = 0; hop < 3; hop++)
        {
            using var message = new HttpRequestMessage(HttpMethod.Get, uri);
            if (uri.Host == "generativelanguage.googleapis.com") GoogleAuth(key)(message.Headers);
            using var response = await client.SendAsync(message, HttpCompletionOption.ResponseHeadersRead, ct);
            if ((int)response.StatusCode is 301 or 302 or 303 or 307 or 308 && response.Headers.Location is { } location)
            {
                uri = location.IsAbsoluteUri ? location : new Uri(uri, location);
                if (!AllowedGoogleDownload(uri)) throw new JsonException("Invalid Google video redirect.");
                continue;
            }
            response.EnsureSuccessStatusCode();
            var mime = response.Content.Headers.ContentType?.MediaType;
            if (mime is not (null or "application/octet-stream") && !mime.StartsWith("video/", StringComparison.OrdinalIgnoreCase))
                throw new JsonException("Google did not return video content.");
            await using var stream = await response.Content.ReadAsStreamAsync(ct);
            return (await BoundedBase64Async(stream, limit, ct), mime is null or "application/octet-stream" ? "video/mp4" : mime);
        }
        throw new JsonException("Too many Google video redirects.");
    }

    private static bool AllowedGoogleDownload(Uri uri) => uri.Scheme == "https" && uri.IsDefaultPort && uri.UserInfo.Length == 0
        && uri.Host is "generativelanguage.googleapis.com" or "storage.googleapis.com" or "www.googleapis.com"
        && (uri.Host != "generativelanguage.googleapis.com" || uri.AbsolutePath.StartsWith("/v1beta/files/", StringComparison.Ordinal)
            || uri.AbsolutePath.StartsWith("/download/v1beta/files/", StringComparison.Ordinal));
}
