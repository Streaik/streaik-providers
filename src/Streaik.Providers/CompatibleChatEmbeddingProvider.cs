using System.Runtime.CompilerServices;
using System.Text.Json;
using Streaik.ChatCompletions.Models;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal abstract class CompatibleChatEmbeddingProvider(IHttpClientFactory clients, IProviderCredentials credentials, string id, string prefix)
    : CompatibleChatProvider(clients, credentials, id, prefix), INativeChatOptionsProvider, IEmbeddingMetadataProvider
{
    public override async Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        PrepareCompatibleRequest(request);
        return await NativeHttp.PostAsync<Dictionary<string, JsonElement>, ChatCompletion>(client, CompatiblePath("chat/completions"),
            NativeOptionsPayload.Build(request, ProviderId, Json, false), Bearer(ApiKey()), headers, Json, ct);
    }

    public override async IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        PrepareCompatibleRequest(request);
        await foreach (var part in NativeHttp.StreamAsync<Dictionary<string, JsonElement>, ChatCompletionUpdate>(client, CompatiblePath("chat/completions"),
            NativeOptionsPayload.Build(request, ProviderId, Json, true), Bearer(ApiKey()), headers, Json,
            part => part.AdditionalProperties?.ContainsKey("error") == true, ct)) yield return part;
    }

    public async Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => (await EmbedWithMetadataAsync(request, headers, ct)).Result;

    public async Task<(OpenAIEmbeddingResponse Result, JsonElement Body, Dictionary<string, string?> Headers)> EmbedWithMetadataAsync(
        OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, CompatiblePath("embeddings"), NativeOptionsPayload.Build(request, ProviderId, Json),
            Bearer(ApiKey()), headers, Json, ct);
        if (Property(body, "data").ValueKind != JsonValueKind.Array) throw new JsonException($"Missing {ProviderId} embeddings.");
        var result = body.Deserialize<OpenAIEmbeddingResponse>(Json) ?? throw new JsonException($"Missing {ProviderId} embedding response.");
        result.Model = String(body, "model") ?? request.Model;
        return (result, body, NativeHttp.SafeResponseMetadataHeaders(upstream));
    }
}
