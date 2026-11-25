using System.Buffers.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class TensorBlockProvider
{
    private const int ImageLimit = 20 * 1024 * 1024;

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        ValidateStreamCount(request.Stream, request.N);
        return ImagesAsync("generations", ImageBody(request), request.OutputFormat, request.Size, request.Quality, request.Background, headers, ct);
    }

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        ValidateStreamCount(request.Stream, request.N);
        var body = ImageBody(request);
        body.Remove("images");
        body.Remove("mask"); // The upstream takes a single seed image, not an OpenAI mask.
        if (request.ImageFiles?.FirstOrDefault() is { } file)
        {
            await using var input = file.OpenReadStream();
            var mime = ImageFormat(file.ContentType) is null ? "image/png" : file.ContentType;
            body["image"] = $"data:{mime};base64,{await BoundedBase64Async(input, ImageLimit, ct)}";
        }
        else if (request.Images?.FirstOrDefault() is { } reference && (reference.ImageUrl ?? reference.FileId) is { Length: > 0 } source)
            body["image"] = source;
        else throw new ProviderRequestException("TensorBlock image edits require an input image.");
        return await ImagesAsync("edits", body, request.OutputFormat, request.Size, request.Quality, request.Background, headers, ct);
    }

    private static void ValidateStreamCount(bool? stream, int? count)
    {
        // The gateway wraps the buffered result in its existing single-completion SSE format.
        if (stream == true && count > 1)
            throw new ProviderRequestException("Image streaming supports one image; use a non-streaming request for multiple images.");
    }

    private static JsonObject ImageBody<T>(T request)
    {
        var body = new JsonObject();
        foreach (var (name, value) in NativeOptionsPayload.Build(request, Id, Json)) body[name] = Copy(value);
        body.Remove("stream");
        body.Remove("partial_images");
        return body;
    }

    private async Task<OpenAIImagesResponse> ImagesAsync(string operation, JsonObject payload, string? format, string? size,
        string? quality, string? background, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        var (raw, upstreamHeaders) = await NativeHttp.PostJsonAsync(client, $"images/{operation}", payload, Auth(), headers, Json, ct);
        if (Property(raw, "data").ValueKind != JsonValueKind.Array) throw new JsonException("Missing TensorBlock image data.");
        var result = raw.Deserialize<OpenAIImagesResponse>(Json) ?? throw new JsonException("Missing TensorBlock image result.");
        var safe = ToObject(raw);
        var safeImages = safe["data"]!.AsArray();
        var remaining = ImageLimit;
        var index = 0;
        foreach (var item in Array(Property(raw, "data")))
        {
            var image = result.Data![index] ?? throw new JsonException("Invalid TensorBlock image item.");
            var detected = ImageFormat(String(item, "mime_type"));
            if (string.IsNullOrWhiteSpace(image.B64Json))
                (image.B64Json, detected) = await DownloadImageAsync(String(item, "url"), remaining, ct);
            if (!Base64.IsValid(image.B64Json, out var bytes) || bytes == 0 || bytes > remaining)
                throw new JsonException("TensorBlock returned invalid or oversized image data.");
            remaining -= bytes;
            result.OutputFormat ??= detected;
#pragma warning disable CS0618 // Normalize upstream URLs to inline images.
            image.Url = null;
#pragma warning restore CS0618
            var safeImage = safeImages[index++]!.AsObject();
            safeImage.Remove("b64_json");
            safeImage.Remove("url");
        }
        if (index == 0) throw new JsonException("TensorBlock did not return usable images.");
        if (result.Created == 0) result.Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        result.OutputFormat ??= format ?? "png";
        result.Size ??= size;
        result.Quality ??= quality;
        result.Background ??= background;
        var metadataHeaders = NativeHttp.SafeResponseMetadataHeaders(upstreamHeaders);
        if (metadataHeaders.Count > 0) safe["responseHeaders"] = JsonSerializer.SerializeToNode(metadataHeaders, Json);
        (result.AdditionalProperties ??= new())["providerMetadata"] = JsonSerializer.SerializeToElement(new Dictionary<string, JsonObject> { [Id] = safe }, Json);
        return result;
    }

    private async Task<(string Base64, string Format)> DownloadImageAsync(string? value, int limit, CancellationToken ct)
    {
        if (!Uri.TryCreate(value, UriKind.Absolute, out var url) || url.Scheme != Uri.UriSchemeHttps || !url.IsDefaultPort
            || url.UserInfo.Length != 0 || url.Fragment.Length != 0 || url.HostNameType != UriHostNameType.Dns || !IsAssetHost(url.IdnHost))
            throw new JsonException("TensorBlock did not return inline data or a trusted image URL.");
        using var client = CreateClient(AssetClient);
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        using var response = await NativeHttp.SendAsync(client, request, ct);
        var format = ImageFormat(response.Content.Headers.ContentType?.MediaType)
            ?? throw new JsonException("Unsupported TensorBlock image type.");
        await using var input = await response.Content.ReadAsStreamAsync(ct);
        return (await BoundedBase64Async(input, limit, ct), format);
    }

    // Forge can relay image URLs from model vendors. Trust extra CDN hosts only when the gateway operator opts in.
    private bool IsAssetHost(string host) => host == "tensorblock.co" || host.EndsWith(".tensorblock.co", StringComparison.Ordinal)
        || configuration?.GetSection("Providers:tensorblock:AssetHosts").GetChildren()
            .Any(entry => host.Equals(entry.Value?.Trim(), StringComparison.OrdinalIgnoreCase)) == true;

    private static string? ImageFormat(string? mime) => mime?.ToLowerInvariant() switch
    {
        "image/png" => "png", "image/jpeg" => "jpeg", "image/webp" => "webp", "image/gif" => "gif", "image/avif" => "avif", _ => null
    };
}
