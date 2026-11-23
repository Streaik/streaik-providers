using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.VoyageAI;

public static class VoyageAIRegistration
{
    public static IServiceCollection AddVoyageAI(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, VoyageAIProvider.Id, "https://api.voyageai.com/v1/");
        return services.AddProvider<VoyageAIProvider>(VoyageAIProvider.Id);
    }
}

internal sealed class VoyageAIProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IEmbeddingsProvider, IRerankingProvider
{
    internal const string Id = "voyageai";

    // Voyage has no model-list endpoint; keep the same catalog as the legacy provider.
    private static readonly string[] Models = ["rerank-2.5", "rerank-2.5-lite", "voyage-multimodal-3.5", "voyage-multimodal-3",
        "voyage-context-3", "voyage-4-large", "voyage-4", "voyage-4-lite", "voyage-3-large", "voyage-3.5", "voyage-3.5-lite",
        "voyage-code-3", "voyage-finance-2", "voyage-law-2"];

    public IAsyncEnumerable<Model> ListModelsAsync(CancellationToken ct) =>
        (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id)) ? [] : Models)
        .Select(model => new Model { Id = model, Name = model, OwnedBy = Id,
            Type = model.StartsWith("rerank-", StringComparison.Ordinal) ? "reranking" : "embedding" })
        .ToAsyncEnumerable();

    public async Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var contextual = request.Model.StartsWith("voyage-context-", StringComparison.OrdinalIgnoreCase);
        var multimodal = request.Model.StartsWith("voyage-multimodal-", StringComparison.OrdinalIgnoreCase);
        var fields = new JsonObject();
        foreach (var (name, value) in request.AdditionalProperties ?? [])
            if (name.ToLowerInvariant() is not ("model" or "input" or "dimensions" or "encoding_format")) fields[name] = Copy(value);
        fields["model"] = request.Model;
        if (request.Dimensions is { } dimensions && !fields.ContainsKey("output_dimension")) fields["output_dimension"] = dimensions;
        if (string.Equals(request.EncodingFormat, "base64", StringComparison.OrdinalIgnoreCase))
            fields[multimodal ? "output_encoding" : "encoding_format"] = "base64";

        if (multimodal) fields["inputs"] = MultimodalInputs(request.Input);
        else if (contextual)
        {
            if (!fields.ContainsKey("inputs")) fields["inputs"] = ContextualInputs(request.Input);
        }
        else if (request.Input.ValueKind == JsonValueKind.String || IsStrings(request.Input)) fields["input"] = Copy(request.Input);
        else throw new ProviderRequestException("Voyage text embeddings require a string or an array of strings.");

        using var client = clients.CreateClient(Id);
        var root = await NativeHttp.PostAsync<JsonObject, JsonElement>(client,
            multimodal ? "multimodalembeddings" : contextual ? "contextualizedembeddings" : "embeddings",
            fields, Auth(), headers, Json, ct);
        var embeddings = new List<OpenAIEmbeddingData>();
        foreach (var item in Array(Property(root, "data")))
        {
            if (Property(item, "embedding").ValueKind != JsonValueKind.Undefined) Add(item);
            else foreach (var chunk in Array(Property(item, "data"))) Add(chunk);
        }
        var count = Property(Property(root, "usage"), "total_tokens");
        var tokens = count.ValueKind == JsonValueKind.Number && count.TryGetInt32(out var n) ? n : 0;
        return new() { Model = String(root, "model") ?? request.Model, Data = embeddings,
            Usage = new() { PromptTokens = tokens, TotalTokens = tokens } };

        void Add(JsonElement item)
        {
            if (Property(item, "embedding") is { ValueKind: JsonValueKind.Array or JsonValueKind.String } vector)
                embeddings.Add(new() { Index = embeddings.Count, Embedding = vector });
        }
    }

    public async Task<RerankingResponse> RerankAsync(RerankingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var fields = new JsonObject();
        if (request.ProviderOptions?.TryGetValue(Id, out var options) == true)
        {
            if (options.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Voyage provider options must be an object.");
            foreach (var property in options.EnumerateObject())
            {
                var name = property.Name;
                var value = property.Value;
                var normalized = name.ToLowerInvariant();
                if (normalized is "model" or "query" or "documents") continue;
                if (normalized is "return_documents" or "truncation")
                {
                    if (value.ValueKind == JsonValueKind.Null) continue;
                    if (value.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
                        throw new ProviderRequestException($"Voyage '{normalized}' must be a boolean.");
                    fields[normalized] = value.GetBoolean();
                }
                else fields[name] = Copy(value);
            }
        }
        fields["model"] = request.Model;
        fields["query"] = request.Query;
        fields["documents"] = Copy(request.Documents.Values);
        if (request.TopN is { } top) fields["top_k"] = top;
        using var client = clients.CreateClient(Id);
        var (root, responseHeaders) = await NativeHttp.PostJsonAsync(client, "rerank", fields, Auth(), headers, Json, ct);
        return new()
        {
            Ranking = Array(Property(root, "data")).Select(item => new RerankingRanking
            { Index = item.GetProperty("index").GetInt32(), RelevanceScore = item.GetProperty("relevance_score").GetDouble() }).ToArray(),
            Response = new() { Id = String(root, "id"), ModelId = request.Model, Timestamp = DateTime.UtcNow, Body = root,
                Headers = responseHeaders.Where(pair => pair.Value is not null).ToDictionary(pair => pair.Key, pair => pair.Value!, StringComparer.OrdinalIgnoreCase) }
        };
    }

    private Action<System.Net.Http.Headers.HttpRequestHeaders> Auth()
    {
        var key = RequiredKey(credentials, Id);
        return headers => headers.Authorization = new("Bearer", key);
    }

    private static bool IsStrings(JsonElement input) => input.ValueKind == JsonValueKind.Array
        && input.EnumerateArray().All(item => item.ValueKind == JsonValueKind.String);

    private static JsonNode? ContextualInputs(JsonElement input)
    {
        if (input.ValueKind == JsonValueKind.String) return new JsonArray(new JsonArray(Copy(input)));
        if (IsStrings(input)) return new JsonArray(input.EnumerateArray().Select(item => (JsonNode)new JsonArray(Copy(item))).ToArray());
        if (input.ValueKind == JsonValueKind.Array && input.EnumerateArray().All(IsStrings)) return Copy(input);
        throw new ProviderRequestException("Voyage contextual embeddings require text, strings, or nested arrays of strings.");
    }

    private static JsonArray MultimodalInputs(JsonElement input)
    {
        if (input.ValueKind == JsonValueKind.String) return new JsonArray(Document([Text(input.GetString())]));
        if (IsStrings(input)) return new JsonArray(input.EnumerateArray().Select(item => Document([Text(item.GetString())])).ToArray());
        if (input.ValueKind != JsonValueKind.Array) throw new ProviderRequestException("Voyage multimodal embeddings require text or an array of content objects.");
        return new JsonArray(input.EnumerateArray().Select(item => Property(item, "content") is { ValueKind: JsonValueKind.Array } content
            ? Document(content.EnumerateArray().Select(MultimodalPart).ToArray())
            : throw new ProviderRequestException("Each Voyage multimodal input requires a content array.")).ToArray());

        static JsonNode Document(JsonNode?[] parts) => new JsonObject { ["content"] = new JsonArray(parts) };
        static JsonNode Text(string? text) => new JsonObject { ["type"] = "text", ["text"] = text };
    }

    private static JsonNode? MultimodalPart(JsonElement part)
    {
        var type = String(part, "type");
        if (type is "text" or "image_base64" or "video_url" or "video_base64") return Copy(part);
        if (type is not ("image_url" or "input_video")) throw new ProviderRequestException($"Voyage does not support content part '{type}'.");
        var media = Property(part, type);
        var value = media.ValueKind == JsonValueKind.String ? media.GetString() : String(media, "url") ?? String(media, "data");
        if (string.IsNullOrWhiteSpace(value)) throw new ProviderRequestException("Voyage multimodal media requires a URL or data URI.");
        var kind = type == "image_url" ? "image" : "video";
        var key = value.StartsWith("data:", StringComparison.OrdinalIgnoreCase) ? kind + "_base64" : kind + "_url";
        return new JsonObject { ["type"] = key, [key] = value };
    }
}
