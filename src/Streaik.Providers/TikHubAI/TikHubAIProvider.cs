using Microsoft.Extensions.DependencyInjection;
using Streaik.Messages;
using Streaik.Responses;
using Streaik.Responses.Streaming;

namespace Streaik.Providers;

public static class TikHubAIProviderRegistration
{
    public static IServiceCollection AddTikHubAI(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, TikHubAIProvider.Id, "https://ai.tikhub.io/v1/");
        return services.AddProvider<TikHubAIProvider>(TikHubAIProvider.Id, "anthropic-version", "anthropic-beta");
    }
}

internal sealed class TikHubAIProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), IResponsesProvider, IMessagesProvider, INativeChatOptionsProvider
{
    internal const string Id = "tikhubai";
    private readonly BearerTextEndpoints text = new(clients, credentials, Id);

    public Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.CreateResponseAsync(request, headers, ct);

    public IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.StreamResponseAsync(request, headers, ct);

    public Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.CreateMessageAsync(request, headers, ct);

    public IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.StreamMessageAsync(request, headers, ct);
}
