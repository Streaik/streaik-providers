using System.Buffers.Binary;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class IsaacusRegistration
{
    public static IServiceCollection AddIsaacus(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, IsaacusProvider.Id, "https://api.isaacus.com/v1/");
        return services.AddProvider<IsaacusProvider>(IsaacusProvider.Id);
    }
}

internal sealed class IsaacusProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IEmbeddingMetadataProvider, IRerankingProvider
{
    internal const string Id = "isaacus";
    private const int BatchSize = 128;
    private static readonly string[] Models = ["kanon-2-embedder", "kanon-2-reranker", "kanon-universal-classifier"];

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        foreach (var id in Models)
        {
            ct.ThrowIfCancellationRequested();
            yield return new() { Id = id, Name = id, OwnedBy = "Isaacus", Type = id == Models[0] ? "embedding" : "reranking" };
        }
    }

    public async Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => (await EmbedWithMetadataAsync(request, headers, ct)).Result;

    public async Task<(OpenAIEmbeddingResponse Result, JsonElement Body, Dictionary<string, string?> Headers)> EmbedWithMetadataAsync(
        OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var texts = request.Input.ValueKind == JsonValueKind.String ? [request.Input] : Array(request.Input).ToArray();
        ValidateTexts(texts);
        var encoding = request.EncodingFormat?.ToLowerInvariant();
        if (encoding is not (null or "float" or "base64")) throw new ProviderRequestException("Unsupported embedding encoding_format.");
        if (request.User is not null) throw new ProviderRequestException("Isaacus embeddings do not support user.");
        if (request.Dimensions is <= 0) throw new ProviderRequestException("dimensions must be positive.");

        var payload = Options(request.AdditionalProperties, "model", "input", "texts", "encoding_format", "user");
        if (request.Dimensions is { } dimensions) payload["dimensions"] = Element(dimensions);
        if (payload.TryGetValue("dimensions", out var size) && size.ValueKind != JsonValueKind.Null &&
            (size.ValueKind != JsonValueKind.Number || !size.TryGetInt32(out var n) || n <= 0))
            throw new ProviderRequestException("dimensions must be null or a positive integer.");
        payload["model"] = Element(request.Model);
        var data = new OpenAIEmbeddingData[texts.Length];
        var batches = new List<JsonElement>();
        var tokens = 0;
        Dictionary<string, string?> responseHeaders = new();
        using var client = clients.CreateClient(Id);
        var auth = Auth();
        for (var offset = 0; offset < texts.Length; offset += BatchSize)
        {
            ct.ThrowIfCancellationRequested();
            var count = Math.Min(BatchSize, texts.Length - offset);
            payload["texts"] = Element(texts.AsSpan(offset, count).ToArray());
            var (root, upstream) = await NativeHttp.PostJsonAsync(client, "embeddings", payload, auth, headers, Json, ct);
            responseHeaders = upstream;
            batches.Add(root);
            tokens = checked(tokens + InputTokens(root));
            var embeddings = Property(root, "embeddings");
            if (embeddings.ValueKind != JsonValueKind.Array || embeddings.GetArrayLength() != count)
                throw new JsonException("Isaacus did not return one embedding per input.");
            var seen = new HashSet<int>();
            foreach (var item in embeddings.EnumerateArray())
            {
                var index = offset + Index(item, count, seen);
                var vector = Property(item, "embedding");
                if (vector.ValueKind != JsonValueKind.Array || vector.GetArrayLength() == 0)
                    throw new JsonException("Isaacus returned an invalid embedding vector.");
                var values = vector.EnumerateArray().Select(FiniteFloat).ToArray();
                data[index] = new() { Index = index, Embedding = encoding == "base64" ? Element(Base64(values)) : vector };
            }
        }

        return (new() { Model = request.Model, Data = data, Usage = new() { PromptTokens = tokens, TotalTokens = tokens } },
            batches.Count == 1 ? batches[0] : Element(new { batches }), responseHeaders);
    }

    public async Task<RerankingResponse> RerankAsync(RerankingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Documents?.Type != "text" || request.Documents.Values.ValueKind != JsonValueKind.Array)
            throw new ProviderRequestException("Isaacus reranking requires text documents.");
        if (string.IsNullOrWhiteSpace(request.Query) || request.TopN is <= 0)
            throw new ProviderRequestException("A query and positive topN are required when topN is supplied.");
        var texts = request.Documents.Values.EnumerateArray().ToArray();
        ValidateTexts(texts);
        var payload = Options(request.ProviderOptions?.TryGetValue(Id, out var options) == true
            ? OptionFields(options) : null, "model", "query", "texts", "documents");
        payload["model"] = Element(request.Model);
        payload["query"] = Element(request.Query);
        payload["texts"] = request.Documents.Values;
        if (request.TopN is { } top) payload["top_n"] = Element(top);

        using var client = clients.CreateClient(Id);
        var (root, upstream) = await NativeHttp.PostJsonAsync(client, "rerankings", payload, Auth(), headers, Json, ct);
        _ = InputTokens(root);
        var results = Property(root, "results");
        if (results.ValueKind != JsonValueKind.Array) throw new JsonException("Isaacus did not return reranking results.");
        var seen = new HashSet<int>();
        return new()
        {
            Ranking = results.EnumerateArray().Select(item => new RerankingRanking
                { Index = Index(item, texts.Length, seen), RelevanceScore = FiniteFloat(Property(item, "score")) }).ToArray(),
            ProviderMetadata = new() { [Id] = root },
            Response = new() { Id = String(root, "id"), ModelId = request.Model, Timestamp = DateTime.UtcNow, Body = root,
                Headers = upstream.Where(pair => pair.Value is not null).ToDictionary(pair => pair.Key, pair => pair.Value!, StringComparer.OrdinalIgnoreCase) }
        };
    }

    private static Dictionary<string, JsonElement> OptionFields(JsonElement options)
        => options.ValueKind == JsonValueKind.Object ? options.EnumerateObject().ToDictionary(p => p.Name, p => p.Value)
            : throw new ProviderRequestException("Isaacus provider options must be an object.");

    private static Dictionary<string, JsonElement> Options(Dictionary<string, JsonElement>? options, params string[] reserved)
        => (options ?? []).Where(pair => !reserved.Contains(pair.Key, StringComparer.OrdinalIgnoreCase))
            .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.OrdinalIgnoreCase);

    private static void ValidateTexts(JsonElement[] texts)
    {
        if (texts.Length == 0 || texts.Any(value => value.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(value.GetString())))
            throw new ProviderRequestException("Isaacus requires non-empty strings, not token IDs.");
    }

    private Action<HttpRequestHeaders> Auth()
    {
        var key = RequiredKey(credentials, Id);
        return headers => headers.Authorization = new("Bearer", key);
    }

    private static JsonElement Element<T>(T value) => JsonSerializer.SerializeToElement(value, Json);
    private static int InputTokens(JsonElement root)
        => Property(Property(root, "usage"), "input_tokens") is { ValueKind: JsonValueKind.Number } tokens
            && tokens.TryGetInt32(out var count) && count >= 0 ? count : throw new JsonException("Invalid Isaacus input token count.");

    private static int Index(JsonElement item, int count, HashSet<int> seen)
        => Property(item, "index") is { ValueKind: JsonValueKind.Number } index && index.TryGetInt32(out var n)
            && n >= 0 && n < count && seen.Add(n) ? n : throw new JsonException("Invalid or duplicate Isaacus input index.");

    private static float FiniteFloat(JsonElement value)
        => value.ValueKind == JsonValueKind.Number && value.TryGetSingle(out var n) && float.IsFinite(n)
            ? n : throw new JsonException("Invalid Isaacus numeric value.");

    private static string Base64(float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        for (var i = 0; i < values.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * sizeof(float)), values[i]);
        return Convert.ToBase64String(bytes);
    }
}
