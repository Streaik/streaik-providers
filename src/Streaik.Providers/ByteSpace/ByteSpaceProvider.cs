using Microsoft.Extensions.DependencyInjection;

namespace Streaik.Providers;

public static class ByteSpaceProviderRegistration
{
    public static IServiceCollection AddByteSpace(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, ByteSpaceProvider.Id, "https://api.bytespace.sg/v1/");
        return services.AddProvider<ByteSpaceProvider>(ByteSpaceProvider.Id);
    }
}

internal sealed class ByteSpaceProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), INativeChatOptionsProvider
{
    internal const string Id = "bytespace";
}
