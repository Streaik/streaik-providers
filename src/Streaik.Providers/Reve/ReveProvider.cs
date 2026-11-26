using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.Catalog;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class ReveProviderRegistration
{
    public static IServiceCollection AddReve(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, ReveProvider.Id, "https://api.reve.com/");
        return services.AddProvider<ReveProvider>(ReveProvider.Id);
    }
}

internal sealed class ReveProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IImageProvider, IStreamingImagesProvider, IModelListingProvider
{
    internal const string Id = "reve";
    private const string CurrentVersion = "reve-v2-create@260601";
    private const int MaxReferenceBytes = 16 * 1024 * 1024;
    private static readonly JsonElement Pricing = JsonSerializer.SerializeToElement(ProviderCatalog.ReadPricing(Id), Json);
    private static readonly HashSet<string> Reserved = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "version", "prompt", "n", "size", "seed", "stream", "partial_images", "response_format",
        "image", "images", "files", "mask"
    };

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        await Task.CompletedTask;
        foreach (var model in new[] { "latest", CurrentVersion })
        {
            ct.ThrowIfCancellationRequested();
            yield return new() { Id = model, Name = model, OwnedBy = "Reve", Type = "image" };
        }
    }

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => CreateAsync(request.Model, request.Prompt, request.N, request.Size, false, request.AdditionalProperties, null, headers, ct);

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var references = new List<object>();
        foreach (var file in request.ImageFiles ?? [])
        {
            await using var stream = file.OpenReadStream();
            references.Add(new { data = await BoundedBase64Async(stream, MaxReferenceBytes, ct) });
        }
        foreach (var image in request.Images ?? [])
        {
            if (image.ImageUrl is not { Length: > 0 } source || image.FileId is not null)
                throw new ProviderRequestException("Reve requires image URLs, inline image data, or uploads; it cannot use file IDs.", 422);
            if (source.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
            {
                var comma = source.IndexOf(',');
                if (comma < 0 || comma == source.Length - 1 || !source[..comma].EndsWith(";base64", StringComparison.OrdinalIgnoreCase))
                    throw new ProviderRequestException("Invalid inline image reference.");
                source = source[(comma + 1)..];
            }
            // URLs are passed to Reve, not downloaded by the gateway.
            references.Add(new { data = source });
        }
        if (references.Count == 0) throw new ProviderRequestException("Reve editing requires a reference image.");
        return await CreateAsync(request.Model, request.Prompt, request.N, request.Size, request.Mask is not null || request.MaskFile is not null,
            request.AdditionalProperties, references, headers, ct);
    }

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        yield return NativeMediaOptions.CompletedImage(await GenerateImageAsync(request, headers, ct), edit: false);
    }

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        yield return NativeMediaOptions.CompletedImage(await EditImageAsync(request, headers, ct), edit: true);
    }

    private async Task<OpenAIImagesResponse> CreateAsync(string model, string prompt, int? n, string? size, bool mask,
        Dictionary<string, JsonElement>? extensions, List<object>? references, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(prompt)) throw new ProviderRequestException("Reve requires a prompt.");
        var options = NativeMediaOptions.Read(Id, extensions, Reserved);
        if (options.TryGetValue("postprocessing", out var postprocessing) && postprocessing.ValueKind is not (JsonValueKind.Array or JsonValueKind.Null))
            throw new ProviderRequestException("Reve postprocessing must be an array.");
        var version = string.IsNullOrWhiteSpace(model) || model.Equals("latest", StringComparison.OrdinalIgnoreCase)
            || model.Equals("reve-v2-create@20260601", StringComparison.OrdinalIgnoreCase) ? CurrentVersion : model;
        var payload = options.ToDictionary(pair => pair.Key, pair => (object?)pair.Value, StringComparer.Ordinal);
        payload["prompt"] = prompt;
        payload["version"] = version;
        if (references is not null)
        {
            foreach (var key in payload.Keys.Where(key => key.Equals("references", StringComparison.OrdinalIgnoreCase)).ToArray()) payload.Remove(key);
            payload["references"] = references;
        }
        var keyValue = RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        var raw = await NativeHttp.PostAsync<object, JsonElement>(client, "v2/image/create", payload,
            outgoing => outgoing.TryAddWithoutValidation("X-API-Key", keyValue), headers, Json, ct);
        if (String(raw, "image") is not { Length: > 0 } image) throw new JsonException("Reve did not return an image.");

        var native = raw.EnumerateObject().Where(field => !field.NameEquals("image"))
            .ToDictionary(field => field.Name, field => field.Value);
        var providerMetadata = NativeMediaOptions.Metadata(Id, native);
        var price = Property(Property(Pricing, $"{Id}/{model}"), "pricing");
        if (ProviderPrices.TokenPrice(Property(price, "output")) is > 0 and var perCredit
            && ProviderPrices.TokenPrice(Property(raw, "credits_used")) is >= 0 and var credits)
            providerMetadata["gateway"] = JsonSerializer.SerializeToElement(new { cost = perCredit * credits }, Json);
        var metadata = new Dictionary<string, JsonElement>
        {
            ["providerMetadata"] = JsonSerializer.SerializeToElement(providerMetadata, Json)
        };
        var warnings = new List<object>();
        if (n is > 1) warnings.Add(new { type = "unsupported", feature = "n", details = "Reve returns one image." });
        if (mask) warnings.Add(new { type = "unsupported", feature = "mask" });
        if (!string.IsNullOrWhiteSpace(size)) warnings.Add(new { type = "unsupported", feature = "size" });
        if (extensions?.ContainsKey("seed") == true) warnings.Add(new { type = "unsupported", feature = "seed" });
        if (warnings.Count != 0) metadata["warnings"] = JsonSerializer.SerializeToElement(warnings, Json);
        return new() { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Data = [new() { B64Json = image }], OutputFormat = "png", AdditionalProperties = metadata };
    }
}
