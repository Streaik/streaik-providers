using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.Messages;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class AnyRouterProviderRegistration
{
    public static IServiceCollection AddAnyRouter(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AnyRouterProvider.Id, "https://anyrouter.dev/api/v1/");
        return services.AddProvider<AnyRouterProvider>(AnyRouterProvider.Id, "anthropic-version", "anthropic-beta");
    }
}

internal sealed partial class AnyRouterProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatEmbeddingProvider(clients, credentials, Id, ""), IResponsesProvider, IMessagesProvider, IRerankingProvider, IDecisionsProvider
{
    internal const string Id = "anyrouter";
    private readonly BearerTextEndpoints text = new(clients, credentials, Id);

    public Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.CreateResponseAsync(request, headers, ct);
    public IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.StreamResponseAsync(request, headers, ct);
    public Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.CreateMessageAsync(request, headers, ct);
    public IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.StreamMessageAsync(request, headers, ct);

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (ConfiguredApiKey() is not { Length: > 0 } key || string.IsNullOrWhiteSpace(key)) yield break;
        var body = await FetchModelsAsync(key, ct);
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var item in Array(Property(body, "data")))
        {
            if (String(item, "id") is not { Length: > 0 } id || !seen.Add(id)) continue;
            var type = id.Equals("anyrouter/decision", StringComparison.OrdinalIgnoreCase)
                || id.Equals("typesafe/jev", StringComparison.OrdinalIgnoreCase) || id.StartsWith("typesafe/jev-", StringComparison.OrdinalIgnoreCase)
                ? "decision" : id.Contains("rerank", StringComparison.OrdinalIgnoreCase) ? "rerank" : String(item, "type");
            var context = Property(item, "context_length");
            yield return new Model { Id = id, Name = String(item, "name") ?? id, Type = type ?? "model",
                OwnedBy = String(item, "provider") ?? String(item, "owned_by") ?? Id, Description = String(item, "description"),
                ContextWindow = context.ValueKind == JsonValueKind.Number && context.TryGetInt32(out var size) ? size : null };
        }
        await foreach (var model in StaticModelListing.ListAsync(Id, key, ct))
            if (seen.Add(model.Id)) yield return model;
    }

    public async Task<RerankingResponse> RerankAsync(RerankingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Query) || request.Documents?.Values.ValueKind != JsonValueKind.Array || request.TopN is <= 0)
            throw new ProviderRequestException("AnyRouter reranking requires a query and an array of documents; topN must be positive.");
        var docs = request.Documents.Values;
        var count = docs.GetArrayLength();
        if (count is < 1 or > 1000 || docs.EnumerateArray().Any(doc => doc.ValueKind != JsonValueKind.String
            && (doc.ValueKind != JsonValueKind.Object || doc.EnumerateObject().Count() != 1 || String(doc, "text") is null)))
            throw new ProviderRequestException("AnyRouter accepts 1 to 1,000 strings or objects with only a string text field.");
        var payload = new Dictionary<string, object> { ["model"] = request.Model, ["query"] = request.Query, ["documents"] = docs };
        if (request.TopN is { } top) payload["top_n"] = top;
        if (request.ProviderOptions?.TryGetValue(Id, out var options) == true)
        {
            if (options.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("AnyRouter options must be an object.");
            foreach (var option in options.EnumerateObject())
            {
                if (option.Name is "model" or "query" or "documents" or "top_n" || option.Name.Equals("authorization", StringComparison.OrdinalIgnoreCase))
                    continue;
                if ((option.Name == "return_documents" && option.Value.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
                    || (option.Name == "max_tokens_per_doc" && (option.Value.ValueKind != JsonValueKind.Number || !option.Value.TryGetInt32(out var n) || n < 1))
                    || (option.Name == "session_id" && option.Value.ValueKind != JsonValueKind.String))
                    throw new ProviderRequestException($"Invalid AnyRouter rerank option '{option.Name}'.");
                payload[option.Name] = option.Value;
            }
        }
        using var client = CreateClient();
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, "rerank", payload, Bearer(ApiKey()), headers, Json, ct);
        var results = Property(body, "results");
        if (results.ValueKind != JsonValueKind.Array) throw new JsonException("Missing AnyRouter rerank results.");
        var seen = new HashSet<int>();
        var ranking = results.EnumerateArray().Select(item =>
        {
            var index = Property(item, "index");
            var score = Property(item, "relevance_score");
            if (index.ValueKind != JsonValueKind.Number || !index.TryGetInt32(out var i) || i < 0 || i >= count || !seen.Add(i)
                || score.ValueKind != JsonValueKind.Number || !score.TryGetSingle(out var s) || !float.IsFinite(s) || s is < 0 or > 1)
                throw new JsonException("Invalid AnyRouter rerank index or score.");
            return new RerankingRanking { Index = i, RelevanceScore = s };
        }).OrderByDescending(item => item.RelevanceScore).Take(request.TopN ?? count).ToArray();
        return new() { Ranking = ranking, ProviderMetadata = NativeMediaOptions.Metadata(Id, body),
            Warnings = string.Equals(request.Documents.Type, "text", StringComparison.OrdinalIgnoreCase) ? []
                : [new { type = "unsupported", feature = "documents.type", details = "AnyRouter reranking accepts text documents." }],
            Response = new() { Timestamp = DateTime.UtcNow, Id = String(body, "id"), ModelId = String(body, "model") ?? request.Model,
                Body = body, Headers = NativeHttp.SafeResponseMetadataHeaders(upstream).ToDictionary(p => p.Key, p => p.Value ?? "") } };
    }
}
