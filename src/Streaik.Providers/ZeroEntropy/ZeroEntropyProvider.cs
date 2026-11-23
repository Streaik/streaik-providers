using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.ZeroEntropy;

public static class ZeroEntropyRegistration
{
    public static IServiceCollection AddZeroEntropy(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, ZeroEntropyProvider.Id, "https://api.zeroentropy.dev/v1/models/");
        return services.AddProvider<ZeroEntropyProvider>(ZeroEntropyProvider.Id);
    }
}

internal sealed class ZeroEntropyProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IEmbeddingMetadataProvider, IRerankingProvider
{
    internal const string Id = "zeroentropy";

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        foreach (var id in new[] { "zerank-2", "zerank-1", "zerank-1-small", "zembed-1" })
        {
            ct.ThrowIfCancellationRequested();
            yield return new() { Id = id, Name = id, OwnedBy = "ZeroEntropy", Type = id == "zembed-1" ? "embedding" : "reranking" };
        }
    }

    public async Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => (await EmbedWithMetadataAsync(request, headers, ct)).Result;

    public async Task<(OpenAIEmbeddingResponse Result, JsonElement Body, Dictionary<string, string?> Headers)> EmbedWithMetadataAsync(
        OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Input.ValueKind is not (JsonValueKind.Array or JsonValueKind.String)) throw new ProviderRequestException("'input' is required.");
        var payload = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, value) in request.AdditionalProperties ?? [])
        {
            if (name.ToLowerInvariant() is "model" or "input" or "dimensions" or "encoding_format" or "user") continue;
            payload[name.Equals("input_type", StringComparison.OrdinalIgnoreCase) ? "input_type" : name] = value;
        }
        payload.TryAdd("input_type", "document");
        payload["model"] = request.Model;
        payload["input"] = request.Input;
        if (request.Dimensions is { } dimensions) payload["dimensions"] = dimensions;
        if (!string.IsNullOrWhiteSpace(request.EncodingFormat)) payload["encoding_format"] = request.EncodingFormat;

        using var client = clients.CreateClient(Id);
        var (root, responseHeaders) = await NativeHttp.PostJsonAsync(client, "embed", payload, Auth(), headers, Json, ct);
        var data = Results(root).Select((item, index) => new OpenAIEmbeddingData
        {
            Index = index,
            Embedding = Property(item, "embedding") is { ValueKind: JsonValueKind.Array or JsonValueKind.String } vector
                ? vector : throw new JsonException("ZeroEntropy did not return an embedding.")
        }).ToArray();
        var usage = Property(Property(root, "usage"), "total_tokens");
        var tokens = usage.ValueKind == JsonValueKind.Number && usage.TryGetInt32(out var count) ? count : 0;
        return (new() { Model = request.Model, Data = data, Usage = new() { PromptTokens = tokens, TotalTokens = tokens } }, root, responseHeaders);
    }

    public async Task<RerankingResponse> RerankAsync(RerankingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Documents?.Values is not { ValueKind: JsonValueKind.Array } input || input.GetArrayLength() == 0)
            throw new ProviderRequestException("At least one document is required.");
        var payload = new Dictionary<string, object?>
        {
            ["model"] = request.Model, ["query"] = request.Query,
            ["documents"] = input.EnumerateArray().Select(item => item.ValueKind == JsonValueKind.String ? item.GetString() : item.GetRawText()).ToArray()
        };
        if (request.TopN is { } top) payload["top_n"] = top;
        if (request.ProviderOptions?.TryGetValue(Id, out var options) == true)
        {
            if (options.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("ZeroEntropy provider options must be an object.");
            foreach (var property in options.EnumerateObject())
                if (property.Name.ToLowerInvariant() is not ("model" or "query" or "documents" or "top_n")) payload[property.Name] = property.Value;
        }

        using var client = clients.CreateClient(Id);
        var (root, responseHeaders) = await NativeHttp.PostJsonAsync(client, "rerank", payload, Auth(), headers, Json, ct);
        return new()
        {
            Ranking = Results(root).Select(item => new RerankingRanking
            {
                Index = Property(item, "index") is { ValueKind: JsonValueKind.Number } index && index.TryGetInt32(out var n)
                    ? n : throw new JsonException("ZeroEntropy did not return a ranking index."),
                RelevanceScore = Property(item, "relevance_score") is { ValueKind: JsonValueKind.Number } score && score.TryGetDouble(out var value)
                    ? value : throw new JsonException("ZeroEntropy did not return a relevance score.")
            }).ToArray(),
            Response = new() { Id = String(root, "id"), ModelId = request.Model, Timestamp = DateTime.UtcNow, Body = root,
                Headers = responseHeaders.Where(pair => pair.Value is not null).ToDictionary(pair => pair.Key, pair => pair.Value!, StringComparer.OrdinalIgnoreCase) }
        };
    }

    private static IEnumerable<JsonElement> Results(JsonElement body) => Property(body, "results") is { ValueKind: JsonValueKind.Array } items
        ? items.EnumerateArray() : throw new JsonException("ZeroEntropy did not return a results array.");

    private Action<HttpRequestHeaders> Auth()
    {
        var key = RequiredKey(credentials, Id);
        return headers => headers.Authorization = new("Bearer", key);
    }
}
