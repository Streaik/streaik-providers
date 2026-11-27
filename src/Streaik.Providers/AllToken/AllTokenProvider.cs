using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.DependencyInjection;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class AllTokenProviderRegistration
{
    public static IServiceCollection AddAllToken(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AllTokenProvider.Id, "https://api.alltoken.ai/v1/");
        PublicMediaDelivery.AddClient(services, AllTokenProvider.AssetClient);
        services.AddMemoryCache();
        return services.AddProvider<AllTokenProvider>(AllTokenProvider.Id);
    }
}

internal sealed partial class AllTokenProvider(IHttpClientFactory clients, IProviderCredentials credentials, IMemoryCache cache)
    : CompatibleChatProvider(clients, credentials, Id), INativeChatOptionsProvider, IImageModelRoutingProvider, IStreamingImagesProvider, IVideoProvider
{
    internal const string Id = "alltoken";
    internal const string AssetClient = "alltoken-assets";
    private const int ImageLimit = 16 * 1024 * 1024;
    private const int VideoLimit = 48 * 1024 * 1024;

    protected override async Task<JsonElement> FetchModelsAsync(string key, CancellationToken ct)
    {
        // AllToken's catalog is caller-dependent, and it is also used to route image turns.
        var cacheKey = (Id, "models", Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(key))));
        if (cache.TryGetValue<JsonElement>(cacheKey, out var cached)) return cached;
        var result = await base.FetchModelsAsync(key, ct);
        return cache.Set(cacheKey, result, TimeSpan.FromMinutes(5));
    }

    public async Task<bool> IsImageModelAsync(string model, CancellationToken ct)
    {
        await foreach (var item in ListModelsAsync(ct))
            if (item.Id == model && item.Type is { } type && !type.Equals("model", StringComparison.OrdinalIgnoreCase))
                return type.Equals("image", StringComparison.OrdinalIgnoreCase) || type.Equals("text_to_image", StringComparison.OrdinalIgnoreCase);
        // The standard AllToken listing can omit type. Recognize its common image families.
        var name = model[(model.LastIndexOf('/') + 1)..].ToLowerInvariant();
        return name.StartsWith("gpt-image", StringComparison.Ordinal) || name.StartsWith("dall-e", StringComparison.Ordinal)
            || name.StartsWith("seedream", StringComparison.Ordinal) || name.StartsWith("imagen-", StringComparison.Ordinal)
            || name.StartsWith("flux-", StringComparison.Ordinal) || name.StartsWith("flux.", StringComparison.Ordinal);
    }

    private async Task<(JsonElement Body, Dictionary<string, string?> Headers)> ReadTaskAsync(HttpClient client, string path,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var response = await SendAsync(client, HttpMethod.Get, path, null, "Bearer", ApiKey(), headers, ct);
        return (await NativeHttp.ReadAsync<JsonElement>(response, Json, ct),
            response.Headers.Concat(response.Content.Headers).ToDictionary(pair => pair.Key, pair => (string?)string.Join(", ", pair.Value), StringComparer.OrdinalIgnoreCase));
    }

    private static string TaskPath(string prefix, string? id)
    {
        if (string.IsNullOrWhiteSpace(id) || id is "." or "..") throw new ProviderRequestException("Invalid AllToken task ID.");
        return prefix + Uri.EscapeDataString(id);
    }

    private static string Status(JsonElement body) => String(body, "status")?.ToLowerInvariant() ?? "queued";
    private static bool Failed(string status) => status is "error" or "failed" or "expired" or "cancelled" or "canceled";
}
