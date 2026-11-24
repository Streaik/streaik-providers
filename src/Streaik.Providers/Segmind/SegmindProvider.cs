using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class SegmindProviderRegistration
{
    public static IServiceCollection AddSegmind(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, SegmindProvider.Id, "https://api.segmind.com/");
        return services.AddProvider<SegmindProvider>(SegmindProvider.Id);
    }
}

internal sealed class SegmindProvider(IHttpClientFactory clients, IProviderCredentials credentials) : IModelListingProvider, IImageProvider
{
    internal const string Id = "segmind";
    private const int ImageLimit = 20 * 1024 * 1024;
    private static readonly HashSet<string> GatewayFields = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "prompt", "n", "size", "response_format", "stream", "partial_images", "files", "images", "mask"
    };

    public IAsyncEnumerable<Model> ListModelsAsync(CancellationToken ct)
        => StaticModelListing.ListAsync(Id, credentials.GetApiKey(Id), ct);

    public async Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Model) || request.Model is "." or ".." || request.Model.Contains('/') || request.Model.Contains('\\'))
            throw new ProviderRequestException("Invalid Segmind model ID.");
        var payload = ToObject(request);
        foreach (var name in payload.Select(entry => entry.Key).Where(GatewayFields.Contains).ToArray()) payload.Remove(name);
        if (request.Prompt is not null) payload["prompt"] = request.Prompt;

        var warnings = new List<object>();
        if (request.N is > 1) warnings.Add(new { type = "unsupported", feature = "n" });
        if (request.Size is not null) warnings.Add(new { type = "unsupported", feature = "size",
            details = "Pass the model's native size parameters through provider metadata." });

        using var client = clients.CreateClient(Id);
        using var message = new HttpRequestMessage(HttpMethod.Post, $"v1/{Uri.EscapeDataString(request.Model)}")
        {
            Content = JsonContent.Create(payload, options: Json)
        };
        message.Headers.Add("x-api-key", RequiredKey(credentials, Id));
        NativeHttp.ApplyHeaders(message, headers);
        using var response = await NativeHttp.SendAsync(client, message, ct);
        var mediaType = response.Content.Headers.ContentType?.MediaType?.ToLowerInvariant();
        var format = mediaType switch
        {
            "image/png" => "png", "image/jpeg" => "jpeg", "image/webp" => "webp", "image/gif" => "gif",
            "image/avif" => "avif", "image/bmp" => "bmp", "image/svg+xml" => "svg",
            _ => throw new JsonException("Segmind did not return a supported image content type.")
        };
        await using var image = await response.Content.ReadAsStreamAsync(ct);
        var encoded = await BoundedBase64Async(image, ImageLimit, ct);
        var rawHeaders = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, values) in response.Headers.Concat(response.Content.Headers)) rawHeaders[name] = string.Join(", ", values);
        var safeHeaders = NativeHttp.SafeResponseMetadataHeaders(rawHeaders);
        if (rawHeaders.TryGetValue("x-seed-value", out var seed)) safeHeaders["x-seed-value"] = seed;
        var extensions = new Dictionary<string, JsonElement>
        {
            ["providerMetadata"] = JsonSerializer.SerializeToElement(new Dictionary<string, object> { [Id] = new { headers = safeHeaders } })
        };
        if (warnings.Count > 0) extensions["warnings"] = JsonSerializer.SerializeToElement(warnings);
        return new OpenAIImagesResponse { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), OutputFormat = format,
            Data = [new() { B64Json = encoded }], AdditionalProperties = extensions };
    }

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => throw new ProviderRequestException("Segmind image inputs are model-specific. Pass native inputs through provider metadata on the generation endpoint.", 422);
}
