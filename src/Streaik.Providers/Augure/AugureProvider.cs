using Microsoft.Extensions.DependencyInjection;

namespace Streaik.Providers;

public static class AugureProviderRegistration
{
    public static IServiceCollection AddAugure(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AugureProvider.Id, "https://api.augureai.ca/v1/");
        return services.AddProvider<AugureProvider>(AugureProvider.Id);
    }
}

// Messages, Responses and Vercel chat are adapted by the gateway from Chat Completions.
internal sealed class AugureProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), INativeChatOptionsProvider
{
    internal const string Id = "augure";
}
