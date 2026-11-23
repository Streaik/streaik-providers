using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.BlackForestLabs;

internal sealed partial class BlackForestLabsProvider
{
    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => ImagesAsync(request.Model, request.Prompt, request.N, request.Size, ImageOptions(request, request.AdditionalProperties), [], null, headers, ct);

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var images = new List<string>();
        foreach (var image in request.Images ?? []) images.Add(ImageReference(image));
        foreach (var file in request.ImageFiles ?? [])
        {
            await using var input = file.OpenReadStream();
            images.Add(await BoundedBase64Async(input, ImageLimit, ct));
        }
        string? mask = request.Mask is null ? null : ImageReference(request.Mask);
        if (request.MaskFile is { } maskFile)
        {
            await using var input = maskFile.OpenReadStream();
            mask = await BoundedBase64Async(input, ImageLimit, ct);
        }
        return await ImagesAsync(request.Model, request.Prompt, request.N, request.Size, ImageOptions(request, request.AdditionalProperties), images, mask, headers, ct);
    }

    private static string ImageReference(OpenAIImageReference image) => image.FileId is not null
        ? throw new ProviderRequestException("Black Forest Labs does not accept file IDs.") : Reference(image.ImageUrl);

    private static (JsonObject Payload, JsonObject Standard) ImageOptions<T>(T request, Dictionary<string, JsonElement>? extra)
    {
        var typed = ToObject(request);
        var native = new JsonObject();
        foreach (var (key, value) in extra ?? [])
        {
            typed.Remove(key);
            if (key.ToLowerInvariant() is not ("model" or "prompt" or "n" or "size" or "stream" or "partial_images" or "response_format"
                or "files" or "images" or "mask" or "image" or "input_image")
                && !key.StartsWith("input_image_", StringComparison.OrdinalIgnoreCase)) native[key] = Copy(value);
        }
        return (native, typed);
    }

    private async Task<OpenAIImagesResponse> ImagesAsync(string model, string prompt, int? n, string? size,
        (JsonObject Payload, JsonObject Standard) options, List<string> images, string? mask, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        model = model.Trim().ToLowerInvariant();
        if (model == "latest") model = "flux-2-pro";
        if (!ImageModels.Contains(model)) throw new ProviderRequestException("Unsupported Black Forest Labs image model.");
        var count = n ?? 1;
        if (count is < 1 or > 10) throw new ProviderRequestException("'n' must be between 1 and 10.");
        var (payload, standard) = options;
        var warnings = new List<object>();
        var flux3 = model is "flux-3-image" or "flux-3-image-klein";
        var tool = model.StartsWith("flux-tools/", StringComparison.Ordinal);
        var erase = model == "flux-tools/erase-v1";
        var outpaint = model == "flux-tools/outpainting-v1";

        foreach (var (key, value) in standard)
        {
            if (value is null || key is "model" or "prompt" or "n" or "size" or "images" or "mask" or "stream" or "partial_images" or "response_format") continue;
            if (key == "output_format" && model != "flux-3-image" || key == "user" && flux3) payload[key] = value.DeepClone();
            else warnings.Add(Warning(key));
        }
        if (mask is not null && !erase) warnings.Add(Warning("mask"));

        if (!tool || outpaint)
        {
            if (!tool && string.IsNullOrWhiteSpace(prompt)) throw new ProviderRequestException("'prompt' is required.");
            if (!string.IsNullOrWhiteSpace(prompt)) payload["prompt"] = prompt;
            var aspect = ReadString(payload["aspect_ratio"]);
            var dimensions = Dimensions(size, aspect, flux3, outpaint);
            if (dimensions is { } d) { payload["width"] = d.Width; payload["height"] = d.Height; }
            if (!flux3) payload.Remove("aspect_ratio");
        }
        if (tool)
        {
            if (images.Count == 0) throw new ProviderRequestException("An input image is required for this FLUX tool.");
            payload[outpaint ? "input_image" : "image"] = images[0];
            if (erase) payload["mask"] = mask ?? throw new ProviderRequestException("A mask is required for erase.");
            if (images.Count > 1) warnings.Add(Warning("files", "Only the first image is used."));
        }
        else if (flux3)
        {
            if (images.Count > 10) throw new ProviderRequestException("FLUX 3 Image supports at most 10 reference images.");
            if (images.Count > 0) payload["images"] = JsonSerializer.SerializeToNode(images, Json);
        }
        else
        {
            var max = model.StartsWith("flux-2-klein-", StringComparison.Ordinal) ? 4 : 8;
            if (images.Count > max) warnings.Add(Warning("files", $"Only the first {max} images are used."));
            for (var i = 0; i < Math.Min(images.Count, max); i++) payload[i == 0 ? "input_image" : $"input_image_{i + 1}"] = images[i];
        }
        var fallback = ReadString(payload["output_format"])?.ToLowerInvariant() switch
        {
            "png" => "image/png", "webp" => "image/webp", "jpeg" or "jpg" => "image/jpeg",
            _ => model == "flux-3-image" ? "image/png" : "image/jpeg"
        };

        using var client = clients.CreateClient(Id);
        var auth = Auth();
        var result = new OpenAIImagesResponse { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Data = [] };
        var metadata = new List<JsonElement>();
        decimal? cost = null;
        for (var i = 0; i < count; i++)
        {
            var (submit, _) = await NativeHttp.PostJsonAsync(client, $"v1/{model}", payload, auth, headers, Json, ct);
            var (_, target) = PollTarget(submit);
            using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
            deadline.CancelAfter(TimeSpan.FromMinutes(10));
            JsonElement final;
            try
            {
                while (true)
                {
                    var (status, body) = await PollAsync(client, target, auth, deadline.Token);
                    if (Ready(status)) { final = body; break; }
                    if (Failed(status)) throw new IOException($"Black Forest Labs image task failed: {status}.");
                    await Task.Delay(TimeSpan.FromSeconds(2), deadline.Token);
                }
            }
            catch (OperationCanceledException ex) when (!ct.IsCancellationRequested)
            { throw new IOException("Black Forest Labs image task timed out.", ex); }

            var media = Property(final, "result");
            if (media.ValueKind == JsonValueKind.Undefined) media = final;
            var start = result.Data.Count;
            foreach (var value in ImageSamples(media))
            {
                var (data, mime) = !value.StartsWith("data:", StringComparison.OrdinalIgnoreCase) && Uri.TryCreate(value, UriKind.Absolute, out var url)
                    ? await DownloadAsync(url, ImageMime(url, fallback), ImageLimit, ct) : Inline(value, fallback, ImageLimit);
                result.Data.Add(new() { B64Json = data });
                result.OutputFormat ??= mime[6..];
            }
            if (result.Data.Count == start) throw new IOException("Black Forest Labs did not return an image.");
            if ((Cost(final) ?? Cost(submit)) is { } amount) cost = (cost ?? 0m) + amount;
            metadata.Add(JsonSerializer.SerializeToElement(new { submit, result = WithoutMedia(final, ImageMediaFields) }, Json));
        }

        var raw = count == 1 ? metadata[0] : JsonSerializer.SerializeToElement(new { results = metadata }, Json);
        result.AdditionalProperties = new() { ["providerMetadata"] = JsonSerializer.SerializeToElement(Metadata(raw, cost), Json) };
        if (cost is { } total) result.AdditionalProperties["cost"] = JsonSerializer.SerializeToElement(total, Json);
        if (warnings.Count != 0) result.AdditionalProperties["warnings"] = JsonSerializer.SerializeToElement(warnings, Json);
        return result;
    }

    private static (int Width, int Height)? Dimensions(string? size, string? aspect, bool flux3, bool outpaint)
    {
        if (!string.IsNullOrWhiteSpace(size) && !size.Equals("auto", StringComparison.OrdinalIgnoreCase))
            return ParsePair(size, 'x', ':') ?? throw new ProviderRequestException("'size' must contain a width and height, or be auto.");
        if (outpaint) throw new ProviderRequestException("Outpainting requires a size containing width and height.");
        if (flux3 || ParsePair(aspect, ':', 'x') is not { } ratio) return null;
        var width = (int)Math.Clamp(1024d * Math.Sqrt((double)ratio.Width / ratio.Height), 64, 2048);
        var height = (int)Math.Clamp(1024d * Math.Sqrt((double)ratio.Height / ratio.Width), 64, 2048);
        return (width, height);
    }

    private static (int Width, int Height)? ParsePair(string? text, char first, char second)
        => text?.ToLowerInvariant().Split(first, second) is [var w, var h]
            && int.TryParse(w, NumberStyles.None, CultureInfo.InvariantCulture, out var width) && width > 0
            && int.TryParse(h, NumberStyles.None, CultureInfo.InvariantCulture, out var height) && height > 0 ? (width, height) : null;

    private static string? ReadString(JsonNode? node) => node is JsonValue value && value.TryGetValue<string>(out var text) ? text : null;

    private static IEnumerable<string> ImageSamples(JsonElement media)
    {
        switch (media.ValueKind)
        {
            case JsonValueKind.String when media.GetString() is { Length: > 0 } value: yield return value; break;
            case JsonValueKind.Array:
                foreach (var item in media.EnumerateArray())
                    foreach (var sample in ImageSamples(item)) yield return sample;
                break;
            case JsonValueKind.Object:
                foreach (var field in ImageMediaFields)
                    if (media.TryGetProperty(field, out var child))
                        foreach (var sample in ImageSamples(child)) yield return sample;
                break;
        }
    }

    private static string ImageMime(Uri uri, string fallback) => Path.GetExtension(uri.AbsolutePath).ToLowerInvariant() switch
    {
        ".png" => "image/png", ".jpg" or ".jpeg" => "image/jpeg", ".webp" => "image/webp", ".gif" => "image/gif", ".avif" => "image/avif", _ => fallback
    };
}
