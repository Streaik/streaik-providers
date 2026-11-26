using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class LeonardoProviderRegistration
{
    public static IServiceCollection AddLeonardo(this IServiceCollection services)
    {
        const string address = "https://cloud.leonardo.ai/api/rest/v1/";
        ProviderServiceCollectionExtensions.AddClient(services, LeonardoProvider.Id, address);
        ProviderServiceCollectionExtensions.AddClient(services, LeonardoProvider.AssetClient, address);
        return services.AddProvider<LeonardoProvider>(LeonardoProvider.Id);
    }
}

internal sealed class LeonardoProvider(IHttpClientFactory clients, IProviderCredentials credentials, IConfiguration configuration)
    : IModelListingProvider, IImageModelRoutingProvider, IStreamingImagesProvider
{
    internal const string Id = "leonardo";
    internal const string AssetClient = "leonardo-assets";
    private const int ImageLimit = 20 * 1024 * 1024;
    private static readonly HashSet<string> Reserved = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "modelId", "model_id", "prompt", "num_images", "n", "size", "aspectRatio", "aspect_ratio",
        "stream", "partial_images", "response_format", "output_format", "image", "images", "mask", "files"
    };

    public Task<bool> IsImageModelAsync(string model, CancellationToken ct) => Task.FromResult(!string.IsNullOrWhiteSpace(model));

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var key = credentials.GetApiKey(Id);
        if (string.IsNullOrWhiteSpace(key)) yield break;
        using var client = clients.CreateClient(Id);
        var response = await NativeHttp.GetAsync<JsonElement>(client, "platformModels", h => h.Authorization = new("Bearer", key), ct);
        foreach (var model in Array(Property(response, "custom_models")))
            if (String(model, "id") is { Length: > 0 } id)
                yield return new() { Id = id, Name = String(model, "name") ?? id, OwnedBy = Id, Type = "image", Description = String(model, "description") };
    }

    public async Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Model) || string.IsNullOrWhiteSpace(request.Prompt) || request.N is <= 0)
            throw new ProviderRequestException("Leonardo requires a model, a prompt, and a positive number of images.");
        var options = NativeMediaOptions.Read(Id, request.AdditionalProperties, Reserved);
        var payload = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, value) in options) payload[name] = value;
        payload["modelId"] = request.Model;
        payload["prompt"] = request.Prompt;
        if (request.N is { } n) payload["num_images"] = n;
        var warnings = new List<object>();
        if (request.Size is { Length: > 0 } size)
        {
            var dimensions = size.Split('x', StringSplitOptions.TrimEntries);
            if (dimensions.Length == 2 && PositiveInt(dimensions[0], out var width) && PositiveInt(dimensions[1], out var height))
            {
                payload.Remove("width");
                payload.Remove("height");
                payload["width"] = width;
                payload["height"] = height;
            }
            else if (size != "auto") warnings.Add(new { type = "unsupported", feature = "size" });
        }
        if (request.AdditionalProperties?.GetValueOrDefault("aspect_ratio").ValueKind == JsonValueKind.String)
            warnings.Add(new { type = "unsupported", feature = "aspectRatio" });

        using var client = clients.CreateClient(Id);
        var key = RequiredKey(credentials, Id);
        var created = await NativeHttp.PostAsync<Dictionary<string, object?>, JsonElement>(client, "generations", payload,
            h => h.Authorization = new("Bearer", key), headers, Json, ct);
        var job = Property(created, "sdGenerationJob");
        if (job.ValueKind != JsonValueKind.Object) job = Property(created, "sd_generation_job");
        var id = String(job, "generationId") ?? String(job, "generation_id");
        if (string.IsNullOrWhiteSpace(id) || id is "." or "..") throw new JsonException("Missing Leonardo generation ID.");

        var path = $"generations/{Uri.EscapeDataString(id)}";
        JsonElement final;
        JsonElement generation;
        for (var attempt = 0; ; attempt++)
        {
            final = await NativeHttp.GetAsync<JsonElement>(client, path, h => h.Authorization = new("Bearer", key), ct);
            generation = Property(final, "generations_by_pk");
            var status = String(generation, "status")?.Trim().ToUpperInvariant();
            if (status == "COMPLETE") break;
            if (status == "FAILED") throw new HttpRequestException("Leonardo image generation failed.");
            if (status != "PENDING") throw new JsonException("Unrecognized Leonardo image generation status.");
            if (attempt >= 90) throw new HttpRequestException("Leonardo image generation did not complete in time.");
            await Task.Delay(TimeSpan.FromSeconds(2), ct);
        }

        var data = new List<OpenAIImageData>();
        string? outputFormat = null;
        var remaining = ImageLimit;
        using var assets = clients.CreateClient(AssetClient);
        foreach (var image in Array(Property(generation, "generated_images")))
        {
            if (String(image, "url") is not { Length: > 0 } url) continue;
            var (base64, format) = await DownloadAsync(assets, url, remaining, ct);
            remaining -= base64.Length / 4 * 3 - (base64.EndsWith("==") ? 2 : base64.EndsWith('=') ? 1 : 0);
            outputFormat ??= format;
            data.Add(new() { B64Json = base64 });
        }
        if (data.Count == 0) throw new JsonException("Leonardo completed without images.");

        // Inline images are already returned above; don't duplicate signed delivery URLs in metadata.
        var raw = ToObject(final);
        if (raw["generations_by_pk"]?["generated_images"] is JsonArray images)
            foreach (var image in images.OfType<JsonObject>()) image.Remove("url");
        var result = new OpenAIImagesResponse
        {
            Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Data = data, OutputFormat = outputFormat,
            AdditionalProperties = new()
            {
                ["providerMetadata"] = JsonSerializer.SerializeToElement(NativeMediaOptions.Metadata(Id, new { generation = created, result = raw }), Json)
            }
        };
        if (warnings.Count > 0) result.AdditionalProperties["warnings"] = JsonSerializer.SerializeToElement(warnings, Json);
        return result;
    }

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => throw new ProviderRequestException("Leonardo image editing is not supported.", 422);

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        if (request.N is not (null or 1)) throw new ProviderRequestException("Leonardo image streaming requires one image.");
        yield return NativeMediaOptions.CompletedImage(await GenerateImageAsync(request, headers, ct), edit: false);
    }

    public IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => throw new ProviderRequestException("Leonardo image editing is not supported.", 422);

    private async Task<(string Base64, string Format)> DownloadAsync(HttpClient client, string source, int limit, CancellationToken ct)
    {
        if (!Uri.TryCreate(source, UriKind.Absolute, out var uri) || uri.Scheme != Uri.UriSchemeHttps || !uri.IsDefaultPort
            || uri.UserInfo.Length != 0 || uri.Fragment.Length != 0 || uri.HostNameType != UriHostNameType.Dns || !IsAssetHost(uri.IdnHost))
            throw new JsonException("Untrusted Leonardo image URL.");

        using var request = new HttpRequestMessage(HttpMethod.Get, uri);
        using var response = await NativeHttp.SendAsync(client, request, ct);
        var format = response.Content.Headers.ContentType?.MediaType?.ToLowerInvariant() switch
        {
            "image/jpeg" or "image/jpg" => "jpeg", "image/webp" => "webp", "image/gif" => "gif",
            "image/png" or null or "application/octet-stream" => "png", _ => throw new JsonException("Invalid Leonardo image content type.")
        };
        if (response.Content.Headers.ContentLength > limit) throw new JsonException("Oversized Leonardo image download.");
        await using var content = await response.Content.ReadAsStreamAsync(ct);
        return (await BoundedBase64Async(content, limit, ct), format);
    }

    private bool IsAssetHost(string host) => host.EndsWith(".leonardo.ai", StringComparison.Ordinal)
        || configuration.GetSection("Providers:leonardo:AssetHosts").GetChildren()
            .Any(entry => string.Equals(entry.Value, host, StringComparison.OrdinalIgnoreCase));

    private static bool PositiveInt(string value, out int result)
        => int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out result) && result > 0;
}
