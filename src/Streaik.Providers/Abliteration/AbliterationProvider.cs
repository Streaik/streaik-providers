using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.Messages;
using Streaik.Responses;
using Streaik.Responses.Streaming;

namespace Streaik.Providers.Abliteration;

public static class AbliterationProviderRegistration
{
    public static IServiceCollection AddAbliteration(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AbliterationProvider.Id, "https://api.abliteration.ai/");
        return services.AddProvider<AbliterationProvider>(AbliterationProvider.Id, "anthropic-version", "anthropic-beta");
    }
}

internal sealed class AbliterationProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id, "v1/"), IResponsesProvider, IMessagesProvider, INativeChatOptionsProvider
{
    internal const string Id = "abliteration";
    private readonly BearerTextEndpoints text = new(clients, credentials, Id, prefix: "v1/", defaultAnthropicVersion: "2023-06-01");

    protected override void PrepareCompatibleRequest(ChatCompletionOptions request)
    {
        base.PrepareCompatibleRequest(request);
        if (request.Stream == true)
        {
            request.StreamOptions ??= new();
            request.StreamOptions.IncludeUsage = true;
        }
    }

    public Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.CreateResponseAsync(request, headers, ct);

    public IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.StreamResponseAsync(request, headers, ct);

    public Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.CreateMessageAsync(request, headers, ct);

    public IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.StreamMessageAsync(request, headers, ct);
}
