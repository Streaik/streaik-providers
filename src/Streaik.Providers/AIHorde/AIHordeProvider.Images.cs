using System.Buffers.Text;
using System.Globalization;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class AIHordeProvider
{
    private const int ImageLimit = 32 * 1024 * 1024;
    // The upstream's R2 account and the bucket-qualified address for signed image results.
    private const string StorageHost = "a223539ccf6caa2d76459c9727d276e6.r2.cloudflarestorage.com";
    private static readonly HashSet<string> ImageControls = new(StringComparer.OrdinalIgnoreCase)
    {
        Id, "provider_options", "providerOptions", "model", "prompt", "n", "size", "seed", "aspect_ratio", "aspectRatio",
        "stream", "response_format", "partial_images", "action", "images", "files", "mask", "authorization",
        "apikey", "api-key", "api_key", "x-api-key", "headers", "client-agent"
    };

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => RunImageAsync(request.Prompt, request.N, request.Size, request.AdditionalProperties, [], null, ct);

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        List<string> images = [];
        foreach (var reference in request.Images ?? []) images.Add(InlineImage(reference));
        foreach (var file in request.ImageFiles ?? []) images.Add(await UploadAsync(file, ct));
        if (images.Count == 0) throw new ProviderRequestException("AI Horde image editing requires an input image.");
        var mask = request.MaskFile is { } upload ? await UploadAsync(upload, ct)
            : request.Mask is { } referenceMask ? InlineImage(referenceMask) : null;
        return await RunImageAsync(request.Prompt, request.N, request.Size, request.AdditionalProperties, images, mask, ct);
    }

    private async Task<OpenAIImagesResponse> RunImageAsync(string prompt, int? n, string? size,
        Dictionary<string, JsonElement>? extras, IReadOnlyList<string> images, string? mask, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(prompt)) throw new ProviderRequestException("AI Horde requires an image prompt.");
        var body = ImagePayload(prompt, n, size, extras, images, mask, out var warnAspect);
        var key = ApiKey();
        void Auth(HttpRequestHeaders headers)
        {
            headers.Add("apikey", key);
            headers.Add("Client-Agent", "Streaik:1:Streaik");
        }

        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromMinutes(10));
        var token = deadline.Token;
        using var client = CreateClient(ImageClient);
        var submission = await NativeHttp.PostAsync<JsonObject, JsonElement>(client, "generate/async", body, Auth,
            new Dictionary<string, string>(), Json, token);
        if (String(submission, "id") is not { Length: > 0 } id || id is "." or "..")
            throw new JsonException("AI Horde did not return an image job ID.");
        var pathId = Uri.EscapeDataString(id);
        while (true)
        {
            var check = await NativeHttp.GetAsync<JsonElement>(client, $"generate/check/{pathId}", Auth, token);
            if (Property(check, "faulted").ValueKind == JsonValueKind.True) throw new IOException("AI Horde image generation failed.");
            if (Property(check, "done").ValueKind == JsonValueKind.True) break;
            await Task.Delay(TimeSpan.FromSeconds(2), token);
        }
        var status = await NativeHttp.GetAsync<JsonElement>(client, $"generate/status/{pathId}", Auth, token);
        if (Property(status, "faulted").ValueKind == JsonValueKind.True) throw new IOException("AI Horde image generation failed.");

        List<OpenAIImageData> data = [];
        string? format = null;
        var remaining = ImageLimit;
        foreach (var generation in Array(Property(status, "generations")))
        {
            if (String(generation, "state") is "faulted" or "failed" || String(generation, "img") is not { Length: > 0 } output) continue;
            var (base64, outputFormat) = await ReadImageAsync(output, remaining, token);
            if (!Base64.IsValid(base64, out var bytes) || bytes == 0 || bytes > remaining)
                throw new IOException("AI Horde returned invalid or oversized image data.");
            remaining -= bytes;
            format ??= outputFormat;
            if (format != outputFormat) throw new IOException("AI Horde returned mixed image formats.");
            data.Add(new() { B64Json = base64 });
        }
        if (data.Count == 0) throw new IOException("AI Horde image generation returned no usable images.");
        var extensions = new Dictionary<string, JsonElement> { ["providerMetadata"] = JsonSerializer.SerializeToElement(new Dictionary<string, JsonElement> { [Id] = status }) };
        if (warnAspect) extensions["warnings"] = JsonSerializer.SerializeToElement(new[] { new { type = "unsupported", feature = "aspectRatio" } });
        return new() { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Data = data, OutputFormat = format, Size = size, AdditionalProperties = extensions };
    }

    private static JsonObject ImagePayload(string prompt, int? n, string? size, Dictionary<string, JsonElement>? extras,
        IReadOnlyList<string> images, string? mask, out bool warnAspect)
    {
        var body = new JsonObject();
        JsonElement seed = default;
        var aspect = false;
        if (extras is not null)
        {
            foreach (var wrapper in new[] { "provider_options", "providerOptions" })
                if (extras.GetValueOrDefault(wrapper) is { ValueKind: JsonValueKind.Object } providers
                    && providers.TryGetProperty(Id, out var options)) AddScoped(options);
            if (extras.TryGetValue(Id, out var scoped)) AddScoped(scoped);
            foreach (var (name, value) in extras) Add(name, value);
        }
        warnAspect = aspect;

        // aihorde-image only routes within Streaik. Only an explicit native `models` option selects upstream models.
        body["prompt"] = prompt;
        var parameters = body["params"] as JsonObject;
        if (body["params"] is not null && parameters is null) throw new ProviderRequestException("AI Horde params must be an object.");
        parameters ??= new JsonObject();
        if (n is { } count) parameters["n"] = count;
        if (seed.ValueKind == JsonValueKind.Number && seed.TryGetInt64(out var number)) parameters["seed"] = number.ToString(CultureInfo.InvariantCulture);
        else if (seed.ValueKind == JsonValueKind.String) parameters["seed"] = seed.GetString();
        if (!string.IsNullOrWhiteSpace(size))
        {
            var dimensions = size.Split(['x', 'X'], StringSplitOptions.TrimEntries);
            if (dimensions.Length != 2 || !int.TryParse(dimensions[0], NumberStyles.None, CultureInfo.InvariantCulture, out var width) || width <= 0
                || !int.TryParse(dimensions[1], NumberStyles.None, CultureInfo.InvariantCulture, out var height) || height <= 0)
                throw new ProviderRequestException("AI Horde image size must be WIDTHxHEIGHT.");
            parameters["width"] = width;
            parameters["height"] = height;
        }
        if (parameters.Count > 0 && body["params"] is null) body["params"] = parameters;
        if (images.Count > 0)
        {
            body["source_image"] = images[0];
            body["source_processing"] = mask is null ? "img2img" : "inpainting";
            if (mask is not null) body["source_mask"] = mask;
            if (images.Count > 1)
                body["extra_source_images"] = new JsonArray(images.Skip(1).Select(image => (JsonNode)new JsonObject { ["image"] = image, ["strength"] = 1 }).ToArray());
        }
        return body;

        void AddScoped(JsonElement options)
        {
            if (options.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("aihorde options must be an object.");
            foreach (var property in options.EnumerateObject()) Add(property.Name, property.Value);
        }

        void Add(string name, JsonElement value)
        {
            if (name is "aspect_ratio" or "aspectRatio") aspect |= value.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(value.GetString());
            else if (name == "seed") seed = value;
            else if (!ImageControls.Contains(name) && value.ValueKind != JsonValueKind.Undefined) body[name] = Copy(value);
        }
    }

    private static string InlineImage(OpenAIImageReference reference)
    {
        var value = reference.ImageUrl;
        if (value?.StartsWith("data:image/", StringComparison.OrdinalIgnoreCase) == true)
        {
            var comma = value.IndexOf(',');
            value = comma > 0 && value.AsSpan(0, comma).EndsWith(";base64", StringComparison.OrdinalIgnoreCase) ? value[(comma + 1)..] : null;
        }
        if (reference.FileId is not null || value is null || !Base64.IsValid(value, out var count) || count == 0 || count > ImageLimit)
            throw new ProviderRequestException("AI Horde requires inline base64 images or uploads, not remote URLs or file IDs.");
        return value;
    }

    private static async Task<string> UploadAsync(IFormFile file, CancellationToken ct)
    {
        if (file.Length <= 0 || file.Length > ImageLimit) throw new ProviderRequestException("Invalid AI Horde image upload size.");
        await using var stream = file.OpenReadStream();
        return await BoundedBase64Async(stream, ImageLimit, ct);
    }

    private async Task<(string Base64, string Format)> ReadImageAsync(string output, int limit, CancellationToken ct)
    {
        if (Base64.IsValid(output)) return (output, "webp"); // AI Horde's inline results are WebP.
        if (!Uri.TryCreate(output, UriKind.Absolute, out var uri) || uri.Scheme != Uri.UriSchemeHttps || !uri.IsDefaultPort || uri.UserInfo.Length != 0
            || uri.IdnHost is not ("aihorde.net" or StorageHost or "stable-horde." + StorageHost))
            throw new IOException("AI Horde did not return inline data or a trusted image URL.");
        using var client = CreateClient(AssetClient);
        using var request = new HttpRequestMessage(HttpMethod.Get, uri);
        using var response = await NativeHttp.SendAsync(client, request, ct);
        var format = response.Content.Headers.ContentType?.MediaType?.ToLowerInvariant() switch
        {
            "image/png" => "png", "image/jpeg" => "jpeg", "image/webp" => "webp", "image/gif" => "gif", "image/avif" => "avif",
            null or "application/octet-stream" => Path.GetExtension(uri.AbsolutePath).ToLowerInvariant() switch
            { ".png" => "png", ".jpg" or ".jpeg" => "jpeg", ".gif" => "gif", ".avif" => "avif", _ => "webp" },
            _ => throw new IOException("AI Horde did not return an image content type.")
        };
        if (response.Content.Headers.ContentLength > limit) throw new IOException("AI Horde returned oversized image data.");
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        return (await BoundedBase64Async(stream, limit, ct), format);
    }
}
