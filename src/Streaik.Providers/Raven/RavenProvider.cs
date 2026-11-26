using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class RavenProviderRegistration
{
    public static IServiceCollection AddRaven(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, RavenProvider.Id, "https://raven.graphn.ai/api/inference/v1/");
        return services.AddProvider<RavenProvider>(RavenProvider.Id);
    }
}

internal sealed class RavenProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatEmbeddingProvider(clients, credentials, Id, ""), IRerankingProvider
{
    internal const string Id = "raven";
    private static readonly HashSet<string> Reserved = new(StringComparer.OrdinalIgnoreCase)
        { "model", "query", "documents", "top_n", "stream" };

    public async Task<RerankingResponse> RerankAsync(RerankingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Query) || request.TopN is <= 0 ||
            request.Documents?.Values is not { ValueKind: JsonValueKind.Array } input || input.GetArrayLength() == 0)
            throw new ProviderRequestException("Raven requires a query, at least one document, and a positive topN when supplied.");
        var documents = input.EnumerateArray().Select(document => document.ValueKind == JsonValueKind.String
            ? document.GetString()! : String(document, "text")
                ?? throw new ProviderRequestException("Raven documents must be strings or objects with a text field.")).ToArray();
        var options = request.ProviderOptions?.TryGetValue(Id, out var native) == true
            ? new Dictionary<string, JsonElement> { [Id] = native } : null;
        var payload = NativeMediaOptions.Read(Id, options, Reserved);
        payload["model"] = JsonSerializer.SerializeToElement(request.Model);
        payload["query"] = JsonSerializer.SerializeToElement(request.Query);
        payload["documents"] = JsonSerializer.SerializeToElement(documents);
        if (request.TopN is { } top) payload["top_n"] = JsonSerializer.SerializeToElement(top);

        using var client = CreateClient();
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, "rerank", payload, Bearer(ApiKey()), headers, Json, ct);
        var results = body.ValueKind == JsonValueKind.Array ? body : Property(body, "results");
        if (results.ValueKind != JsonValueKind.Array) throw new JsonException("Raven did not return a results array.");
        return new()
        {
            Ranking = results.EnumerateArray().Select(Rank).ToList(),
            ProviderMetadata = NativeMediaOptions.Metadata(Id, body),
            Response = new() { Id = String(body, "id"), Timestamp = DateTime.UtcNow, ModelId = request.Model, Body = body,
                Headers = NativeHttp.SafeResponseMetadataHeaders(upstream).Where(x => x.Value is not null)
                    .ToDictionary(x => x.Key, x => x.Value!, StringComparer.OrdinalIgnoreCase) }
        };
    }

    private static RerankingRanking Rank(JsonElement item)
    {
        var index = Property(item, "index");
        var score = Property(item, "relevance_score");
        if (score.ValueKind == JsonValueKind.Undefined) score = Property(item, "score");
        if (index.ValueKind != JsonValueKind.Number || !index.TryGetInt32(out var i)
            || score.ValueKind != JsonValueKind.Number || !score.TryGetSingle(out var s) || !float.IsFinite(s))
            throw new JsonException("Raven returned a ranking without a valid index or score.");
        return new() { Index = i, RelevanceScore = s };
    }
}
