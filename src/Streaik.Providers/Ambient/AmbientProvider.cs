using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.Messages;
using Streaik.Responses;
using Streaik.Responses.Streaming;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class AmbientProviderRegistration
{
    public static IServiceCollection AddAmbient(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AmbientProvider.Id, "https://api.ambient.xyz/v1/");
        return services.AddProvider<AmbientProvider>(AmbientProvider.Id, "anthropic-version", "anthropic-beta");
    }
}

internal sealed class AmbientProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), IResponsesProvider, IMessagesProvider, INativeChatOptionsProvider
{
    internal const string Id = "ambient";
    private readonly BearerTextEndpoints _text = new(clients, credentials, Id);

    public Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => _text.CreateResponseAsync(request, headers, ct);

    public IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => _text.StreamResponseAsync(request, headers, ct);

    public Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => _text.CreateMessageAsync(request, headers, ct);

    public IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => _text.StreamMessageAsync(request, headers, ct);

    public override async IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        PrepareCompatibleRequest(request);
        var body = NativeOptionsPayload.Build(request, Id, Json, true);
        // Ambient can leave the connection open after finish_reason, with no [DONE] marker.
        // Keep reading to preserve the empty-choices usage trailer if it arrives.
        var finishedChoices = new HashSet<int>();
        var expectedChoices = Math.Max(1, request.NumberOfChoices ?? 1);
        await foreach (var part in NativeHttp.StreamAsync<Dictionary<string, JsonElement>, ChatCompletionUpdate>(
            client, ChatEndpointPath, body, ChatAuth(), headers, Json,
            part => part.AdditionalProperties?.ContainsKey("error") == true
                || finishedChoices.Count >= expectedChoices && part.Usage is not null && !part.Choices.Any(),
            ct, endOnKeepaliveOrEof: () => finishedChoices.Count >= expectedChoices))
        {
            if (part.Created >= 1_000_000_000_000L) part.Created /= 1000;
            else if (part.Created == 0) part.Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            foreach (var choice in part.Choices)
                if (!string.IsNullOrWhiteSpace(choice.FinishReason)) finishedChoices.Add(choice.Index);
            yield return part;
        }
    }
}
