using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.Messages;
using Streaik.OpenAI.Models;
using Streaik.OpenAI.Models.Decisions;
using Streaik.Responses;
using Streaik.Responses.Streaming;
using Streaik.Vercel.Models;
using Streaik.Providers.Decisions;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Fireworks;

public static class FireworksRegistration
{
    public static IServiceCollection AddFireworks(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, FireworksProvider.Id, "https://api.fireworks.ai/inference/");
        return services.AddProvider<FireworksProvider>(FireworksProvider.Id, "anthropic-version", "anthropic-beta");
    }
}

public sealed class FireworksProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IDecisionsProvider, IModelListingProvider, IChatCompletionsProvider, IResponsesProvider, IMessagesProvider,
      IEmbeddingsProvider, IRerankingProvider
{
    public const string Id = "fireworks";
    private const string DecisionsPath = "v1/decisions";
    private const string Rune = "accounts/fireworks/models/rune";

    public Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PostAsync<ChatCompletionOptions, ChatCompletion>("v1/chat/completions", PrepareChat(request), headers, Json, ct);

    public IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamAsync<ChatCompletionOptions, ChatCompletionUpdate>("v1/chat/completions", PrepareChat(request), headers, Json, _ => false, ct);

    public Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PostAsync<ResponseRequest, ResponseResult>("v1/responses", request, headers, ResponseJson.Default, ct);

    public IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamAsync<ResponseRequest, ResponseStreamPart>("v1/responses", request, headers, ResponseJson.Default,
            part => part.Type is "response.completed" or "response.incomplete" or "response.failed" or "error", ct);

    public Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PostAsync<MessagesRequest, MessagesResponse>("v1/messages", request, headers, MessagesJson.Default, ct);

    public IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamAsync<MessagesRequest, MessageStreamPart>("v1/messages", request, headers, MessagesJson.Default,
            part => part.Type is "message_stop" or "error", ct);

    public Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PostAsync<OpenAIEmbeddingRequest, OpenAIEmbeddingResponse>("v1/embeddings", request, headers, Json, ct);

    public async Task<RerankingResponse> RerankAsync(RerankingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var documents = request.Documents.Values;
        if (request.Documents.Type != "text" || documents.ValueKind != JsonValueKind.Array || documents.GetArrayLength() == 0
            || Array(documents).Any(x => x.ValueKind != JsonValueKind.String) || request.TopN is <= 0)
            throw new ProviderRequestException("Fireworks reranking requires nonempty text documents and a positive topN when supplied.", 422);
        var payload = new JsonObject();
        if (request.ProviderOptions?.TryGetValue(Id, out var options) == true && options.ValueKind == JsonValueKind.Object)
            foreach (var item in options.EnumerateObject()) payload[item.Name] = Copy(item.Value);
        payload["model"] = request.Model;
        payload["query"] = request.Query;
        payload["documents"] = Copy(documents);
        payload.Remove("top_n");
        if (request.TopN is { } top) payload["top_n"] = top;
        var result = await PostAsync<JsonObject, JsonElement>("v1/rerank", payload, headers, Json, ct);
        var data = Property(result, "data");
        if (data.ValueKind != JsonValueKind.Array) throw new JsonException("Missing Fireworks reranking data.");
        var ranking = Array(data).Select(item =>
        {
            var index = Property(item, "index"); var score = Property(item, "relevance_score");
            if (index.ValueKind != JsonValueKind.Number || !index.TryGetInt32(out var i)
                || i < 0 || i >= documents.GetArrayLength() || score.ValueKind != JsonValueKind.Number || !score.TryGetDouble(out var value))
                throw new JsonException("Invalid Fireworks reranking result.");
            return new RerankingRanking { Index = i, RelevanceScore = value };
        }).OrderByDescending(x => x.RelevanceScore).Take(request.TopN ?? int.MaxValue).ToList();
        return new() { Ranking = ranking, Response = new() { Id = String(result, "id"), Timestamp = DateTime.UtcNow } };
    }

    private static ChatCompletionOptions PrepareChat(ChatCompletionOptions request)
    {
        if (request.Store == false) request.Store = null;
        if (request.Tools?.Any() == true) request.ToolChoice ??= "auto";
        return request;
    }

    private Action<HttpRequestHeaders> Auth()
    {
        var key = RequiredKey(credentials, Id);
        return headers => headers.Authorization = new("Bearer", key);
    }

    private async Task<TResult> PostAsync<TRequest, TResult>(string path, TRequest request, IReadOnlyDictionary<string, string> headers,
        JsonSerializerOptions json, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        return await NativeHttp.PostAsync<TRequest, TResult>(client, path, request, Auth(), headers, json, ct);
    }

    private async IAsyncEnumerable<TEvent> StreamAsync<TRequest, TEvent>(string path, TRequest request,
        IReadOnlyDictionary<string, string> headers, JsonSerializerOptions json, Func<TEvent, bool> terminal,
        [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        await foreach (var part in NativeHttp.StreamAsync(client, path, request, Auth(), headers, json, terminal, ct)) yield return part;
    }

    public async Task<OpenAIDecisionResponse> DecideAsync(OpenAIDecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        return await OpenAICompatibleDecisions.DecideAsync(client, DecisionsPath, RequiredKey(credentials, Id), request, headers, ct);
    }

    public async Task<DecisionResponse> DecideAsync(DecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        return await OpenAICompatibleDecisions.DecideAsync(client, DecisionsPath, RequiredKey(credentials, Id), Id, request, headers, ct);
    }

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var key = credentials.GetApiKey(Id);
        if (string.IsNullOrWhiteSpace(key)) yield break;
        using var client = clients.CreateClient(Id);
        var page = await NativeHttp.GetAsync<JsonElement>(client, "v1/models", h => h.Authorization = new("Bearer", key), ct);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var entry in Array(Property(page, "data")))
        {
            if (String(entry, "id") is not { Length: > 0 } model
                || Property(entry, "active").ValueKind == JsonValueKind.False || !seen.Add(model)) continue;
            var context = Property(entry, "context_length");
            var name = model.Split('/')[^1];
            var type = String(entry, "type")?.ToLowerInvariant();
            var classified = (model, type, name.ToLowerInvariant()) switch
            {
                (Rune, _, _) or (_, "decision" or "decisions", _) => "decision",
                (_, "embedding" or "embeddings", _) => "embedding",
                (_, "rerank" or "reranking", _) => "reranking",
                (_, null or "model", var n) when n.Contains("rerank") => "reranking",
                (_, null or "model", var n) when n.Contains("embed") || n.StartsWith("bge-") => "embedding",
                (_, null or "model", var n) when n.Contains("whisper") || n.Contains("stable-diffusion") || n.StartsWith("flux-") => "other",
                (_, null or "model" or "chat" or "llm" or "text" or "language", _) => "language",
                _ => "other"
            };
            yield return new Model
            {
                Id = model,
                Name = String(entry, "display_name") ?? String(entry, "name") ?? name,
                Description = String(entry, "description"),
                Created = ProviderModelJson.ReadCreated(entry),
                OwnedBy = String(entry, "owned_by") ?? Id,
                Type = classified,
                ContextWindow = context.ValueKind == JsonValueKind.Number && context.TryGetInt32(out var size) ? size : null
            };
        }
    }
}
