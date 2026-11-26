using System.Net.Http.Headers;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;

namespace Streaik.Providers;

public static class AivaraProviderRegistration
{
    public static IServiceCollection AddAivara(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, "aivara", "https://developer.aivara-ai.com/api/");
        return services.AddProvider<AivaraProvider>("aivara");
    }
}

internal sealed class AivaraProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, "aivara"), INativeChatOptionsProvider
{
    protected override string ChatEndpointPath => "v1/ai-gateway";

    protected override Action<HttpRequestHeaders> ChatAuth()
    {
        var key = ApiKey();
        return headers => headers.Add("X-API-Key", key);
    }

    public override IAsyncEnumerable<Model> ListModelsAsync(CancellationToken ct)
        => StaticModelListing.ListAsync(ProviderId, ConfiguredApiKey(), ct);
}
