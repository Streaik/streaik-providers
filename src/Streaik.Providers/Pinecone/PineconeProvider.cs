using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class PineconeProviderRegistration
{
    public static IServiceCollection AddPinecone(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, PineconeProvider.Id, "https://api.pinecone.io/");
        return services.AddProvider<PineconeProvider>(PineconeProvider.Id);
    }
}

internal sealed class PineconeProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IEmbeddingMetadataProvider, IRerankingProvider
{
    internal const string Id = "pinecone";

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (credentials.GetApiKey(Id) is not { Length: > 0 } key || string.IsNullOrWhiteSpace(key)) yield break;
        using var client = clients.CreateClient(Id);
        var root = await NativeHttp.GetAsync<JsonElement>(client, "models", Authenticate(key), ct);
        foreach (var item in Array(Property(root, "models")))
        {
            if (String(item, "model") is not { Length: > 0 } model) continue;
            var type = String(item, "type");
            var vectorType = String(item, "vector_type");
            if (string.Equals(vectorType, "sparse", StringComparison.OrdinalIgnoreCase)) continue;
            yield return new Model { Id = model, Name = model, OwnedBy = Id, Description = String(item, "short_description"),
                Type = type == "rerank" ? "reranking" : "embedding", ContextWindow = Int(item, "max_sequence_length") };
        }
    }

    public async Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => (await EmbedWithMetadataAsync(request, headers, ct)).Result;

    public async Task<(OpenAIEmbeddingResponse Result, JsonElement Body, Dictionary<string, string?> Headers)> EmbedWithMetadataAsync(
        OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Dimensions is not null || request.EncodingFormat is { } encoding && !encoding.Equals("float", StringComparison.OrdinalIgnoreCase))
            throw new ProviderRequestException("Pinecone embeddings support only float encoding and do not accept dimensions.");
        if (request.Model.Equals("pinecone-sparse-english-v0", StringComparison.OrdinalIgnoreCase))
            throw new ProviderRequestException("Sparse vectors are not supported by the embedding contract.");
        var inputs = request.Input.ValueKind switch
        {
            JsonValueKind.String => new[] { request.Input },
            JsonValueKind.Array => request.Input.EnumerateArray().ToArray(),
            _ => []
        };
        if (inputs.Length == 0 || inputs.Any(item => item.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(item.GetString())))
            throw new ProviderRequestException("Pinecone embeddings require a non-empty string or an array of non-empty strings.");

        var options = new Dictionary<string, JsonElement>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, value) in request.AdditionalProperties ?? [])
            if (value.ValueKind != JsonValueKind.Null) options[name] = value;
        var parameters = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        if (options.Remove("parameters", out var raw))
        {
            if (raw.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Pinecone embedding parameters must be an object.");
            foreach (var field in raw.EnumerateObject()) parameters[field.Name] = field.Value;
        }
        if (options.TryGetValue("inputType", out var inputType) || options.TryGetValue("input_type", out inputType))
            parameters["input_type"] = StringOption(inputType, "input_type");
        if (options.TryGetValue("truncate", out var truncate)) parameters["truncate"] = StringOption(truncate, "truncate");

        var (root, responseHeaders) = await PostAsync("embed", new { model = request.Model, parameters,
            inputs = inputs.Select(item => new { text = item }).ToArray() }, headers, ct);
        var model = String(root, "model");
        var usage = Int(Property(root, "usage"), "total_tokens");
        if (!string.Equals(String(root, "vector_type"), "dense", StringComparison.OrdinalIgnoreCase)
            || string.IsNullOrEmpty(model) || usage is null or < 0)
            throw new JsonException("Pinecone did not return a dense embedding response with usage.");
        var data = Items(root).Select((item, index) =>
        {
            var values = Property(item, "values");
            if (values.ValueKind != JsonValueKind.Array || values.GetArrayLength() == 0 ||
                values.EnumerateArray().Any(value => value.ValueKind != JsonValueKind.Number || !value.TryGetDouble(out var n) || !double.IsFinite(n)))
                throw new JsonException("Pinecone did not return a dense numeric vector.");
            return new OpenAIEmbeddingData { Index = index, Embedding = values };
        }).ToArray();
        if (data.Length != inputs.Length) throw new JsonException("Pinecone did not return one embedding per input.");
        return (new() { Model = model, Data = data, Usage = new() { PromptTokens = usage.Value, TotalTokens = usage.Value } }, root, responseHeaders);
    }

    public async Task<RerankingResponse> RerankAsync(RerankingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Documents?.Values is not { ValueKind: JsonValueKind.Array } documents || documents.GetArrayLength() == 0)
            throw new ProviderRequestException("At least one document is required.");
        var payload = new Dictionary<string, object?> { ["model"] = request.Model, ["query"] = request.Query, ["documents"] = documents };
        if (request.TopN is { } top) payload["top_n"] = top;
        if (request.ProviderOptions?.TryGetValue(Id, out var options) == true && options.ValueKind != JsonValueKind.Null)
        {
            if (options.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Pinecone provider options must be an object.");
            foreach (var field in options.EnumerateObject())
            {
                var (name, value) = (field.Name, field.Value);
                var normalized = name.ToLowerInvariant();
                if (normalized is "model" or "query" or "documents" or "top_n" or "topn") continue;
                if (normalized is "return_documents" or "rank_fields" or "parameters")
                {
                    if (value.ValueKind == JsonValueKind.Null) continue;
                    var valid = normalized switch
                    {
                        "return_documents" => value.ValueKind is JsonValueKind.True or JsonValueKind.False,
                        "rank_fields" => value.ValueKind == JsonValueKind.Array && value.EnumerateArray().All(x => x.ValueKind == JsonValueKind.String),
                        _ => value.ValueKind == JsonValueKind.Object
                    };
                    if (!valid) throw new ProviderRequestException($"Invalid Pinecone option '{name}'.");
                    payload[normalized] = value;
                }
                else payload[name] = value;
            }
        }
        var (root, responseHeaders) = await PostAsync("rerank", payload, headers, ct);
        var usage = Property(root, "usage");
        return new()
        {
            Ranking = Items(root).Select(item => new RerankingRanking
            {
                Index = Int(item, "index") is { } index && index >= 0 && index < documents.GetArrayLength() ? index
                    : throw new JsonException("Pinecone did not return a valid ranking index."),
                RelevanceScore = Property(item, "score") is { ValueKind: JsonValueKind.Number } score && score.TryGetDouble(out var number) && double.IsFinite(number)
                    ? number : throw new JsonException("Pinecone did not return a valid ranking score.")
            }).ToArray(),
            ProviderMetadata = usage.ValueKind == JsonValueKind.Object ? new() { [Id] = JsonSerializer.SerializeToElement(new { usage }, Json) } : null,
            Response = new() { Id = String(root, "id"), ModelId = request.Model, Timestamp = DateTime.UtcNow, Body = root,
                Headers = responseHeaders.Where(pair => pair.Value is not null).ToDictionary(pair => pair.Key, pair => pair.Value!, StringComparer.OrdinalIgnoreCase) }
        };
    }

    private async Task<(JsonElement Body, Dictionary<string, string?> Headers)> PostAsync<T>(string path, T payload,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        var (body, responseHeaders) = await NativeHttp.PostJsonAsync(client, path, payload, Authenticate(RequiredKey(credentials, Id)), headers, Json, ct);
        // Only documented inference/API headers and non-sensitive transport metadata are exposed.
        var safe = responseHeaders.Where(pair => pair.Key.ToLowerInvariant() is "x-pinecone-api-version" or "x-pinecone-request-id"
            or "x-request-id" or "x-pinecone-request-latency-ms" or "content-type" or "date" or "retry-after")
            .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.OrdinalIgnoreCase);
        return (body, safe);
    }

    private static Action<HttpRequestHeaders> Authenticate(string key) => headers =>
    {
        headers.Add("Api-Key", key);
        headers.Add("X-Pinecone-Api-Version", "2026-04");
    };

    private static JsonElement StringOption(JsonElement value, string name)
        => value.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(value.GetString()) ? value
            : throw new ProviderRequestException($"Pinecone '{name}' must be a non-empty string.");

    private static int? Int(JsonElement item, string name)
        => Property(item, name) is { ValueKind: JsonValueKind.Number } value && value.TryGetInt32(out var number) ? number : null;

    private static JsonElement.ArrayEnumerator Items(JsonElement root)
        => Property(root, "data") is { ValueKind: JsonValueKind.Array } data ? data.EnumerateArray()
            : throw new JsonException("Pinecone did not return a data array.");
}
