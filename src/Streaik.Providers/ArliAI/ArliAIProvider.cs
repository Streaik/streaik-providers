using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class ArliAIProviderRegistration
{
    public static IServiceCollection AddArliAI(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, ArliAIProvider.Id, "https://api.arliai.com/v1/");
        services.AddMemoryCache();
        return services.AddProvider<ArliAIProvider>(ArliAIProvider.Id);
    }
}

internal sealed class ArliAIProvider(IHttpClientFactory clients, IProviderCredentials credentials, IMemoryCache cache)
    : CompatibleChatProvider(clients, credentials, Id), INativeChatOptionsProvider, IImageModelRoutingProvider
{
    internal const string Id = "arliai";
    private const int ImageLimit = 25 * 1024 * 1024;

    public override Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Tools?.Any() == false) request.Tools = null!;
        return base.CompleteChatAsync(request, headers, ct);
    }

    public override IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Tools?.Any() == false) request.Tools = null!;
        return base.StreamChatAsync(request, headers, ct);
    }

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        await foreach (var model in base.ListModelsAsync(ct))
            if (seen.Add(model.Id))
            {
                if (model.Type == "model") model.Type = "language";
                yield return model;
            }
        foreach (var model in await ImageModelsAsync(ct))
            if (seen.Add(model)) yield return new Model { Id = model, Name = model, OwnedBy = Id, Type = "image" };
    }

    public async Task<bool> IsImageModelAsync(string model, CancellationToken ct) => (await ImageModelsAsync(ct)).Contains(model);

    private async Task<IReadOnlySet<string>> ImageModelsAsync(CancellationToken ct)
    {
        if (ConfiguredApiKey() is not { } key || string.IsNullOrWhiteSpace(key)) return new HashSet<string>();
        var cacheKey = (Id, "images", Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(key))));
        if (cache.TryGetValue<IReadOnlySet<string>>(cacheKey, out var cached) && cached is not null) return cached;
        using var client = CreateClient();
        JsonElement catalog;
        try { catalog = await NativeHttp.GetAsync<JsonElement>(client, "models/image-models", Bearer(key), ct); }
        catch (HttpRequestException ex) when (!ct.IsCancellationRequested && ex.StatusCode is not null)
        {
            // Keep chat usable for keys without image catalog permissions. Retry transient errors sooner.
            return cache.Set(cacheKey, new HashSet<string>(StringComparer.OrdinalIgnoreCase), TimeSpan.FromSeconds(30));
        }
        var values = catalog.ValueKind == JsonValueKind.Array ? catalog : Property(catalog, "data");
        if (values.ValueKind != JsonValueKind.Array) values = Property(catalog, "models");
        var names = Array(values).Select(value => value.ValueKind == JsonValueKind.String ? value.GetString() : new[] { "name", "model_name", "title", "id" }
                .Select(field => String(value, field)).FirstOrDefault(text => !string.IsNullOrWhiteSpace(text)))
            .OfType<string>().Where(text => !string.IsNullOrWhiteSpace(text)).ToHashSet(StringComparer.OrdinalIgnoreCase);
        return cache.Set(cacheKey, names, TimeSpan.FromMinutes(15));
    }

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => ImageAsync("txt2img", Payload(request.Model, request.Prompt, request.N, request.Size, request.AdditionalProperties),
            request.Size, headers, ct);

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var payload = Payload(request.Model, request.Prompt, request.N, request.Size, request.AdditionalProperties);
        if (!payload.ContainsKey("init_images"))
        {
            var images = new JsonArray();
            foreach (var image in request.Images ?? []) images.Add(Inline(image.ImageUrl, input: true));
            foreach (var file in request.ImageFiles ?? []) images.Add(await UploadAsync(file, ct));
            if (images.Count == 0) throw new ProviderRequestException("ArliAI edits require an input image.");
            payload["init_images"] = images;
        }
        if (!payload.ContainsKey("mask"))
        {
            if (request.Mask is { } mask) payload["mask"] = Inline(mask.ImageUrl, input: true);
            else if (request.MaskFile is { } file) payload["mask"] = await UploadAsync(file, ct);
        }
        return await ImageAsync("img2img", payload, request.Size, headers, ct);
    }

    private static JsonObject Payload(string model, string? prompt, int? count, string? size, Dictionary<string, JsonElement>? options)
    {
        var payload = new JsonObject();
        foreach (var (name, value) in options ?? []) payload[name] = Copy(value);
        if (!payload.ContainsKey("sd_model_checkpoint")) payload["sd_model_checkpoint"] = model;
        if (!payload.ContainsKey("prompt")) payload["prompt"] = prompt;
        if (count is not null && !payload.ContainsKey("batch_size")) payload["batch_size"] = count.Value;
        if (size?.Split('x', StringSplitOptions.TrimEntries) is [var width, var height])
        {
            if (!payload.ContainsKey("width") && int.TryParse(width, out var w)) payload["width"] = w;
            if (!payload.ContainsKey("height") && int.TryParse(height, out var h)) payload["height"] = h;
        }
        return payload;
    }

    private async Task<OpenAIImagesResponse> ImageAsync(string endpoint, JsonObject payload, string? size,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        var root = await NativeHttp.PostAsync<JsonObject, JsonElement>(client, endpoint, payload, Bearer(ApiKey()), headers, Json, ct);
        var data = Array(Property(root, "images")).Where(value => value.ValueKind == JsonValueKind.String)
            .Select(value => new OpenAIImageData { B64Json = Inline(value.GetString(), input: false) }).ToList();
        if (data.Count == 0) throw new JsonException("ArliAI returned no images.");
        JsonObject metadata = new();
        foreach (var field in root.EnumerateObject())
            if (field.Name != "images" && field.Name != "parameters") metadata[field.Name] = Copy(field.Value);
        // The upstream may echo input image/mask bytes inside parameters. Keep other values as native JSON.
        if (Property(root, "parameters") is { ValueKind: JsonValueKind.Object } parameters)
            metadata["parameters"] = new JsonObject(parameters.EnumerateObject().Where(field => field.Name is not ("init_images" or "mask"))
                .Select(field => KeyValuePair.Create(field.Name, Copy(field.Value))));
        else if (root.TryGetProperty("parameters", out var raw)) metadata["parameters"] = Copy(raw);
        return new() { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Data = data, Size = size, OutputFormat = "png",
            AdditionalProperties = new() { ["providerMetadata"] = JsonSerializer.SerializeToElement(new JsonObject { [Id] = metadata }, Json) } };
    }

    private static async Task<string> UploadAsync(IFormFile file, CancellationToken ct)
    {
        await using var stream = file.OpenReadStream();
        return await BoundedBase64Async(stream, ImageLimit, ct);
    }

    private static string Inline(string? value, bool input)
    {
        if (value?.StartsWith("data:image/", StringComparison.OrdinalIgnoreCase) == true)
        {
            var comma = value.IndexOf(',');
            if (comma is > 12 and < 64 && value[..comma].EndsWith(";base64", StringComparison.OrdinalIgnoreCase)) value = value[(comma + 1)..];
        }
        if (value is { Length: > 0 and <= ((ImageLimit + 2) / 3) * 4 })
        {
            try { if (Convert.FromBase64String(value).Length is > 0 and <= ImageLimit) return value; }
            catch (FormatException) { }
        }
        if (input) throw new ProviderRequestException("ArliAI requires uploaded or inline base64 images; remote URLs are not fetched.", 422);
        throw new JsonException("ArliAI returned an invalid inline image.");
    }
}
