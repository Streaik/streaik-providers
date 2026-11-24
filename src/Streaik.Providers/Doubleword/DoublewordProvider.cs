using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.Messages;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class DoublewordProviderRegistration
{
    public static IServiceCollection AddDoubleword(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, DoublewordProvider.Id, "https://api.doubleword.ai/v1/");
        return services.AddProvider<DoublewordProvider>(DoublewordProvider.Id, "anthropic-version", "anthropic-beta");
    }
}

internal sealed class DoublewordProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), IResponsesProvider, IMessagesProvider, INativeChatOptionsProvider, IEmbeddingMetadataProvider
{
    internal const string Id = "doubleword";
    private readonly BearerTextEndpoints text = new(clients, credentials, Id);

    public override async Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        PrepareCompatibleRequest(request);
        return await NativeHttp.PostAsync<Dictionary<string, JsonElement>, ChatCompletion>(client, "chat/completions",
            NativeOptionsPayload.Build(request, Id, Json, false), Bearer(ApiKey()), headers, Json, ct);
    }

    public override async IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        PrepareCompatibleRequest(request);
        await foreach (var part in NativeHttp.StreamAsync<Dictionary<string, JsonElement>, ChatCompletionUpdate>(client, "chat/completions",
            NativeOptionsPayload.Build(request, Id, Json, true), Bearer(ApiKey()), headers, Json,
            part => part.AdditionalProperties?.ContainsKey("error") == true, ct)) yield return part;
    }

    public Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.CreateResponseAsync(request, headers, ct);

    public IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.StreamResponseAsync(request, headers, ct);

    public Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.CreateMessageAsync(request, headers, ct);

    public IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.StreamMessageAsync(request, headers, ct);

    public async Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => (await EmbedWithMetadataAsync(request, headers, ct)).Result;

    public async Task<(OpenAIEmbeddingResponse Result, JsonElement Body, Dictionary<string, string?> Headers)> EmbedWithMetadataAsync(
        OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, "embeddings", NativeOptionsPayload.Build(request, Id, Json), Bearer(ApiKey()), headers, Json, ct);
        if (Property(body, "data").ValueKind != JsonValueKind.Array) throw new JsonException("Missing Doubleword embeddings.");
        var result = body.Deserialize<OpenAIEmbeddingResponse>(Json) ?? throw new JsonException("Missing Doubleword embedding response.");
        result.Model = String(body, "model") ?? request.Model;
        return (result, body, NativeHttp.SafeResponseMetadataHeaders(upstream));
    }
}
