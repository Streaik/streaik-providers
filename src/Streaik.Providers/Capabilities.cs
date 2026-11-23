using Streaik.ChatCompletions.Models;
using Streaik.Messages;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;

namespace Streaik.Providers;

// Implement only native capabilities. Each uses its own wire contract.
public interface IChatCompletionsProvider
{
    Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken cancellationToken);
    IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken cancellationToken);
}

// Optional, provider-internal backend for translated chat protocols when it differs from the native Chat Completions route.
public interface IChatBridgeProvider
{
    IChatCompletionsProvider ChatBridge { get; }
}

public interface IResponsesProvider
{
    Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken cancellationToken);
    IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken cancellationToken);
}

public interface IMessagesProvider
{
    Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken cancellationToken);
    IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken cancellationToken);
}

public interface IModelListingProvider
{
    IAsyncEnumerable<Model> ListModelsAsync(CancellationToken cancellationToken);
}

public interface IProviderCredentials
{
    string? GetApiKey(string providerId);
}

public sealed class ProviderCredentialException(string providerId)
    : Exception($"No credentials configured for provider '{providerId}'.");
