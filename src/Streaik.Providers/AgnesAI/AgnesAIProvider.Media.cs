using System.Buffers.Text;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class AgnesAIProvider
{
    private const int ImageLimit = 20 * 1024 * 1024;
    private const int VideoLimit = 50 * 1024 * 1024;
    private static readonly string[] ImageKeys = ["image", "images", "image_url", "imageUrl", "image_urls", "imageUrls"];
    private static readonly HashSet<string> Controls = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "prompt", "stream", "store", "background", "n", "mask", "partial_images", "return_base64", "returnBase64",
        "response_format", "responseFormat", "duration", "fps", "generateAudio", "inputReferences", "frameImages",
        "poll_interval_seconds", "pollIntervalSeconds", "poll_timeout_minutes", "pollTimeoutMinutes", "poll_max_attempts", "pollMaxAttempts"
    };

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => ImageAsync(request.Model, request.Prompt, request.Size, request.N, null, null, false, request.AdditionalProperties, headers, ct);

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => ImageAsync(request.Model, request.Prompt, request.Size, request.N, request.Images, request.ImageFiles,
            request.Mask is not null || request.MaskFile is not null, request.AdditionalProperties, headers, ct);

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        yield return NativeMediaOptions.CompletedImage(await GenerateImageAsync(request, headers, ct), false);
    }

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        yield return NativeMediaOptions.CompletedImage(await EditImageAsync(request, headers, ct), true);
    }

    private async Task<OpenAIImagesResponse> ImageAsync(string model, string prompt, string? size, int? count, OpenAIImageReference[]? references,
        Microsoft.AspNetCore.Http.IFormFile[]? files, bool mask, IReadOnlyDictionary<string, JsonElement>? options,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var warnings = new List<object>();
        Warn(warnings, count is > 1, "n");
        Warn(warnings, mask, "mask");
        var (body, extra, images) = MediaPayload(model, prompt, options);
        var remaining = ImageLimit;
        foreach (var upload in files ?? [])
        {
            await using var input = upload.OpenReadStream();
            var encoded = await BoundedBase64Async(input, remaining, ct);
            remaining -= checked((int)upload.Length);
            var mime = upload.ContentType is "image/png" or "image/jpeg" or "image/webp" or "image/gif" ? upload.ContentType : "image/png";
            images.Add($"data:{mime};base64,{encoded}");
        }
        images.AddRange((references ?? []).Select(reference => reference.ImageUrl).OfType<string>());
        var validImages = images.Where(value => IsWebUrl(value) || value.StartsWith("data:image/", StringComparison.OrdinalIgnoreCase))
            .Distinct(StringComparer.Ordinal).ToArray();
        if (images.Count > 0 && validImages.Length == 0) throw new ProviderRequestException("AgnesAI requires an image URL or inline image.", 422);
        if (validImages.Length > 0) extra["image"] = JsonSerializer.SerializeToNode(validImages, Json);
        var ratio = TakeText(body, "aspectRatio") ?? TakeText(body, "aspect_ratio");
        if (ratio is not null) body["ratio"] = ratio;
        if (size is null && InferSize(ratio ?? Text(body["ratio"])) is { } inferred) size = $"{inferred.Width}x{inferred.Height}";
        if (!string.IsNullOrWhiteSpace(size)) body["size"] = size;
        body["return_base64"] = true;
        extra["response_format"] = "b64_json";
        body["extra_body"] = extra;

        using var client = CreateClient();
        var (raw, _) = await NativeHttp.PostJsonAsync(client, "v1/images/generations", body, Bearer(ApiKey()), headers, Json, ct);
        var output = new List<OpenAIImageData>();
        var outputLimit = ImageLimit;
        string? format = null;
        foreach (var item in Array(Property(raw, "data")))
        {
            var data = String(item, "b64_json");
            var mime = "image/png";
            if (data is { Length: > 0 } && data.StartsWith("data:image/", StringComparison.OrdinalIgnoreCase))
            {
                var comma = data.IndexOf(',');
                if (comma < 0 || !data[..comma].EndsWith(";base64", StringComparison.OrdinalIgnoreCase))
                    throw new JsonException("AgnesAI returned an invalid image data URL.");
                mime = data[5..(comma - 7)];
                data = data[(comma + 1)..];
            }
            else if (data is not { Length: > 0 } && String(item, "url") is { Length: > 0 } url)
                (data, mime) = await DownloadAsync(url, "image/png", outputLimit, ct);
            if (data is null || !Base64.IsValid(data, out var length) || length == 0 || length > outputLimit)
                throw new JsonException("AgnesAI returned missing, invalid, or oversized image data.");
            outputLimit -= length;
            format ??= mime switch { "image/jpeg" => "jpeg", "image/webp" => "webp", "image/gif" => "gif", _ => "png" };
            output.Add(new() { B64Json = data, RevisedPrompt = String(item, "revised_prompt") });
        }
        if (output.Count == 0) throw new JsonException("AgnesAI returned no images.");
        var extensions = new Dictionary<string, JsonElement> { ["providerMetadata"] = JsonSerializer.SerializeToElement(NativeMediaOptions.Metadata(Id, raw), Json) };
        if (warnings.Count > 0) extensions["warnings"] = JsonSerializer.SerializeToElement(warnings, Json);
        return new() { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), OutputFormat = format, Data = [.. output], AdditionalProperties = extensions };
    }

    public async Task<VideoOperationStartResult> StartVideoAsync(VideoRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var options = request.ProviderOptions?.TryGetValue(Id, out var native) == true
            ? new Dictionary<string, JsonElement> { [Id] = native } : null;
        var (body, extra, images) = MediaPayload(request.Model, request.Prompt, options);
        var warnings = new List<object>();
        Warn(warnings, request.Duration is not null, "duration");
        Warn(warnings, request.N is not null, "n");
        Warn(warnings, request.GenerateAudio is not null, "generateAudio");
        if (request.Seed is { } seed) body["seed"] = seed;
        if (request.Fps is { } fps) body["frame_rate"] = fps;
        var resolution = TakeText(body, "resolution");
        var size = TakeText(body, "size");
        var hasNativeSize = body.ContainsKey("width") && body.ContainsKey("height");
        var dimensions = Dimensions(request.Resolution) ?? (!hasNativeSize
            ? Dimensions(size ?? resolution) ?? InferSize(request.AspectRatio) : null);
        if (dimensions is { } value) { body["width"] = value.Width; body["height"] = value.Height; }

        var inputs = new[] { request.Image }.Concat(request.InputReferences ?? []).Concat(request.FrameImages?.Select(frame => frame.Image) ?? [])
            .OfType<VideoFile>().ToArray();
        images.InsertRange(0, inputs.OfType<VideoFileUrl>().Select(image => image.Url));
        var urls = images.Where(IsWebUrl).Distinct(StringComparer.Ordinal).ToArray();
        if (inputs.Length > 0 && urls.Length == 0) throw new ProviderRequestException("AgnesAI video inputs require public image URLs.", 422);
        Warn(warnings, inputs.Any(input => input is not VideoFileUrl url || !IsWebUrl(url.Url)), "inline image");
        var topMode = TakeText(body, "mode");
        var mode = TakeText(extra, "mode") ?? topMode;
        var nested = extra.Count > 0 || urls.Length > 1 || string.Equals(mode, "keyframes", StringComparison.OrdinalIgnoreCase);
        if (urls.Length > 0)
        {
            if (nested) extra["image"] = JsonSerializer.SerializeToNode(urls, Json);
            else body["image"] = urls[0];
        }
        if (mode is not null) (nested ? extra : body)["mode"] = mode;
        if (extra.Count > 0) body["extra_body"] = extra;

        using var client = CreateClient();
        var (raw, upstream) = await NativeHttp.PostJsonAsync(client, "v1/videos", body, Bearer(ApiKey()), headers, Json, ct);
        if (String(raw, "video_id") is not { Length: > 0 and <= 512 } videoId || videoId.Any(char.IsControl))
            throw new JsonException("AgnesAI returned an invalid video ID.");
        var operation = Base64Url.EncodeToString(JsonSerializer.SerializeToUtf8Bytes(new Job(videoId, request.Model), Json));
        return new() { Operation = operation, Warnings = warnings, ProviderMetadata = new() { [Id] = raw }, Response = Info(request.Model, upstream) };
    }

    public async Task<VideoOperationStatusResult> GetVideoAsync(string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        Job job;
        try { job = JsonSerializer.Deserialize<Job>(Base64Url.DecodeFromChars(operation), Json) ?? throw new JsonException(); }
        catch (Exception error) when (error is FormatException or JsonException) { throw new ProviderRequestException("Invalid AgnesAI video operation."); }
        if (string.IsNullOrWhiteSpace(job.VideoId) || string.IsNullOrWhiteSpace(job.Model)) throw new ProviderRequestException("Invalid AgnesAI video operation.");
        using var client = CreateClient();
        using var response = await SendAsync(client, HttpMethod.Get, $"agnesapi?video_id={Uri.EscapeDataString(job.VideoId)}", null, "Bearer", ApiKey(), headers, ct);
        var raw = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        var upstream = response.Headers.Concat(response.Content.Headers).GroupBy(pair => pair.Key, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => (string?)string.Join(", ", group.SelectMany(pair => pair.Value)), StringComparer.OrdinalIgnoreCase);
        VideoOperationStatusResult status;
        switch (String(raw, "status")?.ToLowerInvariant())
        {
            case "failed" or "error" or "canceled" or "cancelled":
                status = new VideoOperationErrorResult { Error = String(raw, "error") ?? String(raw, "message") ?? "AgnesAI video generation failed." };
                break;
            case "completed":
                var url = OutputUrl(raw) ?? OutputUrl(Property(raw, "metadata"));
                if (url is null) { status = new VideoOperationErrorResult { Error = "AgnesAI returned no video URL." }; break; }
                var (data, mime) = await DownloadAsync(url, "video/mp4", VideoLimit, ct);
                status = new VideoOperationCompletedResult { Videos = [new() { Type = "base64", Data = data, MediaType = mime }] };
                break;
            default:
                status = new VideoOperationPendingResult();
                break;
        }
        status.Response = Info(job.Model, upstream);
        status.ProviderMetadata = new() { [Id] = raw };
        return status;
    }

    private static (JsonObject Body, JsonObject Extra, List<string> Images) MediaPayload(string model, string prompt, IReadOnlyDictionary<string, JsonElement>? options)
    {
        if (string.IsNullOrWhiteSpace(model) || string.IsNullOrWhiteSpace(prompt)) throw new ProviderRequestException("AgnesAI requires a model and prompt.");
        var body = new JsonObject();
        var extra = new JsonObject();
        var images = new List<string>();
        foreach (var (key, value) in NativeMediaOptions.Read(Id, options, Controls))
        {
            if (ImageKeys.Contains(key, StringComparer.OrdinalIgnoreCase)) AddImages(value);
            else if (key is "extra_body" or "extraBody")
            {
                if (value.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("AgnesAI extra_body must be an object.");
                foreach (var (name, field) in NativeMediaOptions.Read(Id, value.EnumerateObject().ToDictionary(prop => prop.Name, prop => prop.Value), Controls))
                    if (ImageKeys.Contains(name, StringComparer.OrdinalIgnoreCase)) AddImages(field);
                    else extra[name] = Copy(field);
            }
            else body[key] = Copy(value);
        }
        body["model"] = model;
        body["prompt"] = prompt;
        return (body, extra, images);

        void AddImages(JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.String) images.Add(element.GetString()!.Trim());
            else foreach (var item in Array(element)) if (item.ValueKind == JsonValueKind.String) images.Add(item.GetString()!.Trim());
        }
    }

    private static bool IsWebUrl(string value) => Uri.TryCreate(value, UriKind.Absolute, out var uri)
        && uri.Scheme is "https" or "http" && uri.UserInfo.Length == 0;

    private static string? OutputUrl(JsonElement value) => String(value, "url") ?? String(value, "video_url") ?? String(value, "videoUrl");
    private static void Warn(List<object> warnings, bool unsupported, string feature) { if (unsupported) warnings.Add(new { type = "unsupported", feature }); }
    private static string? Text(JsonNode? node) => node is JsonValue value && value.TryGetValue<string>(out var text) ? text : null;
    private static string? TakeText(JsonObject body, string name) => body.Remove(name, out var node) ? Text(node) : null;
    private static (int Width, int Height)? Dimensions(string? text) => text?.Split(['x', 'X', ':'], StringSplitOptions.TrimEntries) is [var width, var height]
        && int.TryParse(width, NumberStyles.None, CultureInfo.InvariantCulture, out var w) && int.TryParse(height, NumberStyles.None, CultureInfo.InvariantCulture, out var h)
        && w > 0 && h > 0 ? (w, h) : null;

    private static (int Width, int Height)? InferSize(string? ratio)
    {
        if (Dimensions(ratio) is not { } dims) return null;
        var scale = 1536d / Math.Max(dims.Width, dims.Height);
        return ((int)Math.Clamp(Math.Round(dims.Width * scale / 8) * 8, 256, 1536),
            (int)Math.Clamp(Math.Round(dims.Height * scale / 8) * 8, 256, 1536));
    }

    private static HeaderResponseData Info(string model, Dictionary<string, string?> headers) => new()
        { ModelId = $"{Id}/{model}", Timestamp = DateTime.UtcNow,
            Headers = headers.Where(pair => pair.Value is not null).ToDictionary(pair => pair.Key, pair => pair.Value!, StringComparer.OrdinalIgnoreCase) };

    private async Task<(string Data, string Mime)> DownloadAsync(string url, string fallback, int limit, CancellationToken ct)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) || uri.Scheme != Uri.UriSchemeHttps || !uri.IsDefaultPort
            || uri.HostNameType != UriHostNameType.Dns || uri.UserInfo.Length != 0 || uri.Fragment.Length != 0)
            throw new IOException("AgnesAI returned an untrusted asset URL.");
        using var client = CreateClient(DeliveryClient);
        using var request = new HttpRequestMessage(HttpMethod.Get, uri);
        using var result = await NativeHttp.SendAsync(client, request, ct);
        var mime = result.Content.Headers.ContentType?.MediaType;
        if (mime is null or "application/octet-stream") mime = fallback;
        if (!mime.StartsWith(fallback[..(fallback.IndexOf('/') + 1)], StringComparison.OrdinalIgnoreCase))
            throw new IOException("AgnesAI returned an unexpected asset type.");
        if (result.Content.Headers.ContentLength > limit) throw new IOException("AgnesAI media exceeds the inline size limit.");
        await using var stream = await result.Content.ReadAsStreamAsync(ct);
        return (await BoundedBase64Async(stream, limit, ct), mime.ToLowerInvariant());
    }

    internal static SocketsHttpHandler CreateDeliveryHandler() => new()
    {
        UseCookies = false, AllowAutoRedirect = false, UseProxy = false, ConnectTimeout = TimeSpan.FromSeconds(15),
        // Agnes can return public CDN URLs on different hosts. Validate the actual connection target, not a separate DNS lookup.
        ConnectCallback = async (context, ct) =>
        {
            foreach (var address in (await Dns.GetHostAddressesAsync(context.DnsEndPoint.Host, ct)).Where(IsPublicAddress))
            {
                var socket = new Socket(address.AddressFamily, SocketType.Stream, ProtocolType.Tcp);
                try
                {
                    await socket.ConnectAsync(new IPEndPoint(address, context.DnsEndPoint.Port), ct);
                    return new NetworkStream(socket, ownsSocket: true);
                }
                catch (SocketException) { socket.Dispose(); }
                catch { socket.Dispose(); throw; }
            }
            throw new IOException("Could not connect to a public AgnesAI delivery address.");
        }
    };

    private static bool IsPublicAddress(IPAddress address)
    {
        if (address.IsIPv4MappedToIPv6) address = address.MapToIPv4();
        var b = address.GetAddressBytes();
        return address.AddressFamily == AddressFamily.InterNetwork
            ? b[0] is not (0 or 10 or 127 or >= 224)
                && !(b[0] == 100 && b[1] is >= 64 and <= 127) && !(b[0] == 169 && b[1] == 254)
                && !(b[0] == 172 && b[1] is >= 16 and <= 31) && !(b[0] == 192 && b[1] is 0 or 168)
                && !(b[0] == 198 && b[1] is 18 or 19 or 51) && !(b[0] == 203 && b[1] == 0 && b[2] == 113)
            : (b[0] & 0xE0) == 0x20 && !(b[0] == 0x20 && (b[1] == 2 || b[1] == 1 && (b[2] < 2 || b[2] == 0x0D && b[3] == 0xB8)))
                && !(b[0] == 0x3F && b[1] == 0xFF);
    }

    private sealed record Job(string VideoId, string Model);
}
