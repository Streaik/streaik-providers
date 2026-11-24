using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class ArceeAIProviderRegistration
{
    public static IServiceCollection AddArceeAI(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, ArceeAIProvider.Id, "https://api.arcee.ai/api/v1/");
        services.AddMemoryCache();
        return services.AddProvider<ArceeAIProvider>(ArceeAIProvider.Id);
    }
}

internal sealed class ArceeAIProvider(IHttpClientFactory clients, IProviderCredentials credentials, IMemoryCache cache)
    : CompatibleChatProvider(clients, credentials, Id), INativeChatOptionsProvider
{
    internal const string Id = "arceeai";

    public override Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        request.Store = null;
        return base.CompleteChatAsync(request, headers, ct);
    }

    public override IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        request.Store = null;
        return base.StreamChatAsync(request, headers, ct);
    }

    protected override ModelPricing? Pricing(JsonElement model)
        => ProviderPrices.ReadPair(Property(model, "pricing"), "prompt", "completion");

    protected override async Task<JsonElement> FetchModelsAsync(string key, CancellationToken ct)
    {
        // The model listing doubles as the upstream rate card. Never share it across credentials.
        var cacheKey = (Id, "models", Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(key))));
        if (cache.TryGetValue<JsonElement>(cacheKey, out var page)) return page;
        page = await base.FetchModelsAsync(key, ct);
        return cache.Set(cacheKey, page, TimeSpan.FromMinutes(30));
    }
}
