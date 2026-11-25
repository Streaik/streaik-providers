using Microsoft.Extensions.DependencyInjection;
using Streaik.Messages;

namespace Streaik.Providers;

public static class AichixiaRegistration
{
    public static IServiceCollection AddAichixia(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AichixiaProvider.Id, "https://www.aichixia.xyz/api/v1/");
        return services.AddProvider<AichixiaProvider>(AichixiaProvider.Id, "anthropic-version", "anthropic-beta");
    }
}

internal sealed class AichixiaProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, "aichixia"), IMessagesProvider, INativeChatOptionsProvider
{
    internal const string Id = "aichixia";
    private readonly BearerTextEndpoints _text = new(clients, credentials, Id);

    public Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => _text.CreateMessageAsync(request, headers, ct);

    public IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => _text.StreamMessageAsync(request, headers, ct);
}
