using System.Buffers.Text;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class YollomiProviderRegistration
{
    public static IServiceCollection AddYollomi(this IServiceCollection services)
    {
        const string address = "https://yollomi.com/api/v1/";
        ProviderServiceCollectionExtensions.AddClient(services, YollomiProvider.Id, address);
        ProviderServiceCollectionExtensions.AddClient(services, YollomiProvider.AssetClient, address);
        return services.AddProvider<YollomiProvider>(YollomiProvider.Id);
    }
}

internal sealed class YollomiProvider(IHttpClientFactory clients, IProviderCredentials credentials, IConfiguration configuration)
    : IModelListingProvider, IImageProvider
{
    internal const string Id = "yollomi";
    internal const string AssetClient = "yollomi-assets";
    private const int ImageLimit = 20 * 1024 * 1024;

    private static readonly HashSet<string> Reserved = new(StringComparer.OrdinalIgnoreCase)
    {
        Id, "providerOptions", "providerMetadata", "type", "model", "modelId", "prompt", "stream", "partial_images",
        "response_format", "n", "files", "authorization", "api-key", "api_key", "apiKey", "x-api-key", "headers"
    };

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        // The legacy integration lists the public catalog without a key. Catalog entries are
        // informational; only image execution is implemented by this provider.
        using var client = clients.CreateClient(Id);
        var body = await NativeHttp.GetAsync<JsonElement>(client, "models", _ => { }, ct);
        foreach (var type in new[] { "image", "video" })
            foreach (var item in Array(Property(body, type)))
            {
                if (String(item, "modelId") is not { Length: > 0 } model) continue;
                yield return new() { Id = model, Name = String(item, "displayName") ?? model, OwnedBy = Id,
                    Type = type, Description = String(item, "description") };
            }
    }

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var payload = BuildPayload(request.Model, request.Prompt, request.Size, request.N, request.Stream, request.AdditionalProperties);
        return GenerateAsync(payload, request.OutputFormat, headers, ct);
    }

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var payload = BuildPayload(request.Model, request.Prompt, request.Size, request.N, request.Stream, request.AdditionalProperties);
        var images = new List<string>();
        var remaining = ImageLimit;
        foreach (var file in request.ImageFiles ?? []) images.Add(await UploadAsync(file));
        foreach (var image in request.Images ?? []) images.Add(InputReference(image));
        if (images.Count == 0) throw new ProviderRequestException("Yollomi edits require an input image.");
        payload["imageUrl"] = payload["image"] = images[0];
        payload.Remove("images");
        if (images.Count > 1) payload["images"] = images;
        if (request.MaskFile is { } maskFile) payload["mask"] = await UploadAsync(maskFile);
        else if (request.Mask is { } mask) payload["mask"] = InputReference(mask);
        return await GenerateAsync(payload, request.OutputFormat, headers, ct);

        async Task<string> UploadAsync(IFormFile file)
        {
            if (file.Length <= 0 || file.Length > remaining) throw new ProviderRequestException("Invalid or oversized Yollomi image upload.");
            using var stream = file.OpenReadStream();
            var base64 = await BoundedBase64Async(stream, remaining, ct);
            remaining -= (int)file.Length;
            var mime = ImageFormat(file.ContentType) is null ? "image/png" : file.ContentType;
            return $"data:{mime};base64,{base64}";
        }
    }

    private static string InputReference(OpenAIImageReference image)
    {
        // Remote references are sent to Yollomi, never fetched by the gateway.
        if (image.ImageUrl is not { Length: > 0 } source)
            throw new ProviderRequestException("Yollomi requires an image URL or uploaded bytes, not an OpenAI file ID.");
        if (source.StartsWith("data:", StringComparison.OrdinalIgnoreCase)
            || Uri.TryCreate(source, UriKind.Absolute, out var url) && url.Scheme is "http" or "https") return source;
        if (Base64.IsValid(source, out var bytes) && bytes > 0 && bytes <= ImageLimit) return $"data:image/png;base64,{source}";
        throw new ProviderRequestException("Invalid Yollomi image reference.");
    }

    private static Dictionary<string, object?> BuildPayload(string model, string prompt, string? size, int? count, bool? stream,
        Dictionary<string, JsonElement>? options)
    {
        if (stream == true && count > 1)
            throw new ProviderRequestException("Yollomi image streaming supports one image; use a non-streaming request for multiple images.");
        var payload = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
        if (options?.TryGetValue(Id, out var scoped) == true) AddOptions(scoped);
        if (options?.TryGetValue("providerOptions", out var envelope) == true)
        {
            if (envelope.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("providerOptions must be an object.");
            if (envelope.TryGetProperty(Id, out var native)) AddOptions(native);
        }
        foreach (var (name, value) in options ?? []) AddOption(name, value);
        payload["type"] = "image";
        payload["modelId"] = model;
        payload["prompt"] = prompt;
        if (size is not null) payload["size"] = size;
        if (count.HasValue) payload["numOutputs"] = count.Value;
        return payload;

        void AddOptions(JsonElement native)
        {
            if (native.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Yollomi options must be an object.");
            foreach (var property in native.EnumerateObject()) AddOption(property.Name, property.Value);
        }

        void AddOption(string name, JsonElement value)
        {
            if (Reserved.Contains(name)) return;
            // A case-insensitive dictionary retains the first key spelling, so normalize native mapped keys.
            var key = name.ToLowerInvariant() switch
            {
                "aspect_ratio" or "aspectratio" => "aspectRatio", "numoutputs" => "numOutputs",
                "imageurl" => "imageUrl", "image" => "image", "images" => "images", "mask" => "mask",
                "size" => "size", "seed" => "seed", _ => name
            };
            payload[key] = value;
        }
    }

    private async Task<OpenAIImagesResponse> GenerateAsync(Dictionary<string, object?> payload, string? format,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        var (raw, responseHeaders) = await NativeHttp.PostJsonAsync(client, "generate", payload,
            h => h.Authorization = new("Bearer", RequiredKey(credentials, Id)), headers, Json, ct);
        var images = Array(Property(raw, "images")).Where(item => item.ValueKind == JsonValueKind.String)
            .Select(item => item.GetString()!).Where(value => !string.IsNullOrWhiteSpace(value)).ToArray();
        if (images.Length == 0 && String(raw, "image") is { Length: > 0 } image) images = [image];
        if (images.Length == 0) throw new JsonException("Yollomi did not return usable images.");

        var remaining = ImageLimit;
        var data = new List<OpenAIImageData>(images.Length);
        string? detected = null;
        foreach (var value in images)
        {
            var (base64, mimeFormat) = await ReadImageAsync(value, remaining, ct);
            if (!Base64.IsValid(base64, out var bytes) || bytes == 0 || bytes > remaining)
                throw new JsonException("Yollomi returned invalid or oversized image data.");
            remaining -= bytes;
            detected ??= mimeFormat;
            data.Add(new() { B64Json = base64 });
        }

        // Preserve nested upstream diagnostics, without duplicating the image payload or exposing signed asset URLs.
        var metadata = ToObject(raw);
        metadata.Remove("images");
        metadata.Remove("image");
        var safeHeaders = NativeHttp.SafeResponseMetadataHeaders(responseHeaders);
        if (safeHeaders.Count > 0 && !metadata.ContainsKey("responseHeaders"))
            metadata["responseHeaders"] = JsonSerializer.SerializeToNode(safeHeaders, Json);
        return new()
        {
            Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            Data = data,
            OutputFormat = detected ?? format ?? "png",
            Size = payload.TryGetValue("size", out var size) ? size as string : null,
            AdditionalProperties = new()
            {
                ["providerMetadata"] = JsonSerializer.SerializeToElement(new Dictionary<string, object> { [Id] = metadata }, Json)
            }
        };
    }

    private async Task<(string Base64, string? Format)> ReadImageAsync(string value, int limit, CancellationToken ct)
    {
        if (value.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
        {
            var comma = value.IndexOf(',');
            const string encoding = ";base64";
            if (comma < 5 + encoding.Length || !value.AsSpan(0, comma).EndsWith(encoding, StringComparison.OrdinalIgnoreCase)
                || ImageFormat(value[5..(comma - encoding.Length)]) is not { } format)
                throw new JsonException("Invalid Yollomi image data URL.");
            return (value[(comma + 1)..], format);
        }
        if (!Uri.TryCreate(value, UriKind.Absolute, out var url)) return (value, null);
        if (url.Scheme != Uri.UriSchemeHttps || !url.IsDefaultPort || url.UserInfo.Length != 0 || url.Fragment.Length != 0
            || url.HostNameType != UriHostNameType.Dns || !IsAssetHost(url.IdnHost))
            throw new JsonException("Untrusted Yollomi image URL.");
        using var client = clients.CreateClient(AssetClient);
        using var message = new HttpRequestMessage(HttpMethod.Get, url);
        using var response = await NativeHttp.SendAsync(client, message, ct);
        var mediaType = response.Content.Headers.ContentType?.MediaType;
        var detected = ImageFormat(mediaType);
        if ((detected is null && mediaType is not (null or "application/octet-stream")) || response.Content.Headers.ContentLength > limit)
            throw new JsonException("Invalid or oversized Yollomi image download.");
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        return (await BoundedBase64Async(stream, limit, ct), detected);
    }

    private bool IsAssetHost(string host) => host == "yollomi.com" || host.EndsWith(".yollomi.com", StringComparison.Ordinal)
        || configuration.GetSection("Providers:yollomi:AssetHosts").GetChildren()
            .Any(entry => string.Equals(entry.Value, host, StringComparison.OrdinalIgnoreCase));

    private static string? ImageFormat(string? mime) => mime?.ToLowerInvariant() switch
    {
        "image/png" => "png", "image/jpeg" or "image/jpg" => "jpeg", "image/webp" => "webp", "image/gif" => "gif",
        "image/avif" => "avif", "image/bmp" => "bmp", "image/svg+xml" => "svg", _ => null
    };
}
