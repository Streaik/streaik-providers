using Microsoft.Extensions.DependencyInjection;
using Streaik.Messages;

namespace Streaik.Providers;

public static class BaseAPIProviderRegistration
{
    public static IServiceCollection AddBaseAPI(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, BaseAPIProvider.Id, "https://api.baseapi.dev/v1/");
        return services.AddProvider<BaseAPIProvider>(BaseAPIProvider.Id, "anthropic-version", "anthropic-beta");
    }
}

internal sealed class BaseAPIProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), IMessagesProvider, INativeChatOptionsProvider
{
    internal const string Id = "baseapi";
    private readonly BearerTextEndpoints messages = new(clients, credentials, Id);

    public Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => messages.CreateMessageAsync(request, headers, ct);

    public IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => messages.StreamMessageAsync(request, headers, ct);
}
