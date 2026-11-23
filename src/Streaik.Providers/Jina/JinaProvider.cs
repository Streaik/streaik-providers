using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Jina;

public static class JinaRegistration
{
    public static IServiceCollection AddJina(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, JinaProvider.Id, "https://deepsearch.jina.ai/v1/");
        ProviderServiceCollectionExtensions.AddClient(services, JinaProvider.SearchClient, "https://api.jina.ai/v1/");
        return services.AddProvider<JinaProvider>(JinaProvider.Id);
    }
}

internal sealed class JinaProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), INativeChatOptionsProvider, IEmbeddingMetadataProvider, IRerankingProvider
{
    internal const string Id = "jina";
    internal const string SearchClient = "jina-search";

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        yield return new() { Id = "jina-deepsearch-v1", Name = "jina-deepsearch-v1", OwnedBy = "Jina", Type = "model", Created = 1740787200 };
        foreach (var id in new[] { "jina-reranker-v3", "jina-reranker-m0", "jina-colbert-v2", "jina-reranker-v2-base-multilingual",
            "jina-reranker-v1-tiny-en", "jina-reranker-v1-turbo-en", "jina-reranker-v1-base-en", "jina-colbert-v1-en" })
        {
            ct.ThrowIfCancellationRequested();
            yield return new() { Id = id, Name = id, OwnedBy = "Jina", Type = "reranking" };
        }
        foreach (var (id, context) in new[] { ("jina-embeddings-v3", 8192), ("jina-embeddings-v4", 32768), ("jina-clip-v2", 8192),
            ("jina-embeddings-v5-omni-nano", 8192), ("jina-embeddings-v5-omni-small", 32768),
            ("jina-embeddings-v5-text-nano", 8192), ("jina-embeddings-v5-text-small", 32768) })
        {
            ct.ThrowIfCancellationRequested();
            yield return new() { Id = id, Name = id, OwnedBy = "Jina", Type = "embedding", ContextWindow = context };
        }
    }

    public async Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => (await EmbedWithMetadataAsync(request, headers, ct)).Result;

    public async Task<(OpenAIEmbeddingResponse Result, JsonElement Body, Dictionary<string, string?> Headers)> EmbedWithMetadataAsync(
        OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!(request.Input.ValueKind == JsonValueKind.Array && request.Input.GetArrayLength() > 0
            || request.Input.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(request.Input.GetString())))
            throw new ProviderRequestException("Nonempty 'input' is required.");
        if (request.Dimensions is <= 0) throw new ProviderRequestException("'dimensions' must be positive.");
        var encoding = request.EncodingFormat?.ToLowerInvariant();
        if (encoding is not (null or "float" or "base64")) throw new ProviderRequestException("'encoding_format' must be float or base64.");

        var payload = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
        foreach (var (key, value) in request.AdditionalProperties ?? [])
        {
            var name = key.ToLowerInvariant();
            if (name is "model" or "input" or "dimensions" or "encoding_format" or "user" || name == "embedding_type" && encoding is not null) continue;
            payload[name == "embedding_type" ? name : key] = value;
        }
        payload["model"] = request.Model;
        payload["input"] = request.Input;
        if (request.Dimensions is { } dimensions) payload["dimensions"] = dimensions;
        if (encoding is not null) payload["embedding_type"] = encoding;

        using var client = clients.CreateClient(SearchClient);
        var (root, responseHeaders) = await NativeHttp.PostJsonAsync(client, "embeddings", payload, Bearer(ApiKey()), headers, Json, ct);
        var data = Items(root, "data").Select((item, index) => new OpenAIEmbeddingData
        {
            Object = String(item, "object") ?? "embedding",
            Index = Int(item, "index") ?? index,
            Embedding = Property(item, "embedding") is { ValueKind: JsonValueKind.Array or JsonValueKind.String } vector
                ? vector : throw new JsonException("Jina did not return an embedding vector.")
        }).ToArray();
        var usage = Property(root, "usage");
        var prompt = Int(usage, "prompt_tokens");
        var total = Int(usage, "total_tokens");
        return (new() { Object = String(root, "object") ?? "list", Model = String(root, "model") ?? request.Model, Data = data,
            Usage = new() { PromptTokens = prompt ?? total ?? 0, TotalTokens = total ?? prompt ?? 0 } }, root, responseHeaders);
    }

    public async Task<RerankingResponse> RerankAsync(RerankingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Documents?.Values is not { ValueKind: JsonValueKind.Array } documents || documents.GetArrayLength() == 0)
            throw new ProviderRequestException("At least one document is required.");
        var payload = new Dictionary<string, object?> { ["model"] = request.Model, ["query"] = request.Query, ["documents"] = documents };
        if (request.TopN is { } top) payload["top_n"] = top;
        if (request.ProviderOptions?.TryGetValue(Id, out var options) == true && options.ValueKind != JsonValueKind.Null)
        {
            if (options.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Jina provider options must be an object.");
            foreach (var property in options.EnumerateObject())
            {
                var name = property.Name.ToLowerInvariant();
                if (name is "model" or "query" or "documents" or "top_n") continue;
                if (name is "return_documents" or "return_embeddings" or "truncation" or "max_doc_length")
                {
                    if (property.Value.ValueKind == JsonValueKind.Null) continue;
                    var valid = name == "max_doc_length"
                        ? property.Value.ValueKind == JsonValueKind.Number && property.Value.TryGetInt32(out _)
                        : property.Value.ValueKind is JsonValueKind.True or JsonValueKind.False;
                    if (!valid) throw new ProviderRequestException($"Invalid Jina option '{name}'.");
                    payload[name] = property.Value;
                }
                else payload[property.Name] = property.Value;
            }
        }

        using var client = clients.CreateClient(SearchClient);
        var (root, responseHeaders) = await NativeHttp.PostJsonAsync(client, "rerank", payload, Bearer(ApiKey()), headers, Json, ct);
        var usage = Property(root, "usage");
        return new()
        {
            Ranking = Items(root, "results").Select(item => new RerankingRanking
            {
                Index = Int(item, "index") ?? throw new JsonException("Jina did not return a ranking index."),
                RelevanceScore = Property(item, "relevance_score") is { ValueKind: JsonValueKind.Number } score && score.TryGetDouble(out var value)
                    ? value : throw new JsonException("Jina did not return a relevance score.")
            }).ToArray(),
            ProviderMetadata = usage.ValueKind == JsonValueKind.Object ? new() { [Id] = JsonSerializer.SerializeToElement(new { usage }, Json) } : null,
            Response = new() { Id = String(root, "id"), ModelId = request.Model, Timestamp = DateTime.UtcNow, Body = root,
                Headers = responseHeaders.Where(pair => pair.Value is not null).ToDictionary(pair => pair.Key, pair => pair.Value!, StringComparer.OrdinalIgnoreCase) }
        };
    }

    private static IEnumerable<JsonElement> Items(JsonElement body, string name) => Property(body, name) is { ValueKind: JsonValueKind.Array } items
        ? items.EnumerateArray() : throw new JsonException($"Jina did not return a '{name}' array.");

    private static int? Int(JsonElement body, string name)
        => Property(body, name) is { ValueKind: JsonValueKind.Number } number && number.TryGetInt32(out var count) ? count : null;
}
