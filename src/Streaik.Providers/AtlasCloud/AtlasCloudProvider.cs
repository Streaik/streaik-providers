using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.AtlasCloud;

public static class AtlasCloudRegistration
{
    public static IServiceCollection AddAtlasCloud(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AtlasCloudProvider.Id, "https://api.atlascloud.ai/");
        ProviderServiceCollectionExtensions.AddClient(services, AtlasCloudProvider.CatalogClient, "https://console.atlascloud.ai/");
        ProviderServiceCollectionExtensions.AddClient(services, AtlasCloudProvider.AssetClient, "https://api.atlascloud.ai/");
        services.AddMemoryCache();
        return services.AddProvider<AtlasCloudProvider>(AtlasCloudProvider.Id);
    }
}

internal sealed partial class AtlasCloudProvider(IHttpClientFactory clients, IProviderCredentials credentials, IMemoryCache cache)
    : CompatibleChatProvider(clients, credentials, Id, "v1/"), INativeChatOptionsProvider, IImageModelRoutingProvider, IVideoProvider
{
    internal const string Id = "atlascloud";
    internal const string CatalogClient = "atlascloud-catalog";
    internal const string AssetClient = "atlascloud-assets";
    private const string CatalogCacheKey = "Streaik.AtlasCloud.Models";

    private Task<JsonElement> CatalogAsync(CancellationToken ct) => cache.GetOrCreateAsync(CatalogCacheKey, async entry =>
    {
        entry.AbsoluteExpirationRelativeToNow = TimeSpan.FromMinutes(10);
        using var client = CreateClient(CatalogClient);
        return await NativeHttp.GetAsync<JsonElement>(client, "api/v1/models",
            headers => headers.UserAgent.Add(new ProductInfoHeaderValue("Streaik", "1.0")), ct);
    });

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(ConfiguredApiKey())) yield break;
        foreach (var entry in Array(Property(await CatalogAsync(ct), "data")))
        {
            ct.ThrowIfCancellationRequested();
            if (String(entry, "model") is not { Length: > 0 } model) continue;
            yield return new()
            {
                Id = model, Name = String(entry, "displayName") ?? model, Type = ModelType(entry),
                OwnedBy = String(entry, "organization") ?? Id, Description = String(entry, "profile"),
                ContextWindow = Int(entry, "contextLength"), MaxTokens = Int(entry, "maxCompletionTokens")
            };
        }
    }

    public async Task<bool> IsImageModelAsync(string model, CancellationToken ct)
    {
        // Require credentials for a provider call, but never attach them to the public console catalog.
        _ = ApiKey();
        return Array(Property(await CatalogAsync(ct), "data")).Any(entry =>
            string.Equals(String(entry, "model"), model, StringComparison.Ordinal) && ModelType(entry) == "image");
    }

    private static int? Int(JsonElement source, string field)
        => Property(source, field) is { ValueKind: JsonValueKind.Number } number && number.TryGetInt32(out var value) ? value : null;

    private static string ModelType(JsonElement entry) => String(entry, "type")?.Trim().ToLowerInvariant() switch
    {
        "text" or "chat" or "llm" => "language",
        "text-to-image" or "image-to-image" => "image",
        "text-to-video" or "image-to-video" => "video",
        { Length: > 0 } type => type,
        _ => "other"
    };
}
