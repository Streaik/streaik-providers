using Microsoft.Extensions.DependencyInjection;

namespace Streaik.Providers.AIMagicx;

public static class AIMagicxProviderRegistration
{
    public static IServiceCollection AddAIMagicx(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AIMagicxProvider.Id, "https://www.aimagicx.com/api/");
        return services.AddProvider<AIMagicxProvider>(AIMagicxProvider.Id);
    }
}

internal sealed class AIMagicxProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id, "v1/"), INativeChatOptionsProvider
{
    internal const string Id = "aimagicx";
}
