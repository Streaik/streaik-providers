using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class CirrascaleProviderRegistration
{
    public static IServiceCollection AddCirrascale(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, CirrascaleProvider.Id, "https://aisuite.cirrascale.com/apis/v2/");
        return services.AddProvider<CirrascaleProvider>(CirrascaleProvider.Id);
    }
}

internal sealed class CirrascaleProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), INativeChatOptionsProvider, IImageProvider, IStreamingImagesProvider, IRerankingProvider
{
    internal const string Id = "cirrascale";
    private static readonly HashSet<string> ImageFields = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "prompt", "n", "size", "response_format", "stream", "partial_images", "aspect_ratio", "aspectRatio", "files", "images", "mask"
    };
    private static readonly HashSet<string> RerankFields = new(StringComparer.OrdinalIgnoreCase) { "model", "query", "documents", "top_n" };

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (ConfiguredApiKey() is not { } key || string.IsNullOrWhiteSpace(key)) yield break;
        var body = await FetchModelsAsync(key, ct);
        foreach (var (category, type) in new[] { ("llm", "language"), ("text_to_image", "image") })
        foreach (var item in Array(Property(body, category)))
            if (item.ValueKind == JsonValueKind.String && item.GetString() is { Length: > 0 } model)
                yield return new() { Id = model, Name = model, Type = type, OwnedBy = "Cirrascale" };
        const string reranker = "BAAI/bge-reranker-base";
        yield return new() { Id = reranker, Name = reranker, Type = "reranking", OwnedBy = "Cirrascale" };
    }

    public async Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var payload = NativeMediaOptions.Read(Id, request.AdditionalProperties, ImageFields);
        payload["model"] = JsonSerializer.SerializeToElement(request.Model);
        payload["prompt"] = JsonSerializer.SerializeToElement(request.Prompt);
        payload["response_format"] = JsonSerializer.SerializeToElement("b64_json");
        if (request.Size is { Length: > 0 } size) payload["size"] = JsonSerializer.SerializeToElement(size);
        if (request.N is { } count) payload["n"] = JsonSerializer.SerializeToElement(count);
        using var client = CreateClient();
        var (body, _) = await NativeHttp.PostJsonAsync(client, "images/generations", payload, Bearer(ApiKey()), headers, Json, ct);
        var result = CompatibleImageGeneration.ReadResult(body, Id, "png");
        if (result.Data!.Any(item => string.IsNullOrWhiteSpace(item.B64Json))) throw new JsonException("Cirrascale returned no inline image.");
        return result;
    }

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => throw new ProviderRequestException("Cirrascale does not support image editing.", 422);

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageGenerationRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        if (request.N is > 1) throw new ProviderRequestException("Image streaming requires a single output.");
        yield return NativeMediaOptions.CompletedImage(await GenerateImageAsync(request, headers, ct), edit: false);
    }

    public IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => throw new ProviderRequestException("Cirrascale does not support image editing.", 422);

    public async Task<RerankingResponse> RerankAsync(RerankingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var payload = NativeMediaOptions.ReadScoped(Id, request.ProviderOptions, RerankFields);
        payload["model"] = JsonSerializer.SerializeToElement(request.Model);
        payload["query"] = JsonSerializer.SerializeToElement(request.Query);
        payload["documents"] = request.Documents.Values;
        if (request.TopN is { } top) payload["top_n"] = JsonSerializer.SerializeToElement(top);
        using var client = CreateClient();
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, "reranker", payload, Bearer(ApiKey()), headers, Json, ct);
        var data = Property(body, "data");
        if (data.ValueKind != JsonValueKind.Array) throw new JsonException("Cirrascale did not return a ranking array.");
        return new()
        {
            Ranking = data.EnumerateArray().Select(Rank).ToList(),
            ProviderMetadata = NativeMediaOptions.Metadata(Id, body),
            Response = new() { Id = String(body, "id"), Timestamp = DateTime.UtcNow, ModelId = request.Model, Body = body,
                Headers = NativeHttp.SafeResponseMetadataHeaders(upstream).ToDictionary(pair => pair.Key, pair => pair.Value ?? "") }
        };
    }

    private static RerankingRanking Rank(JsonElement item)
    {
        var index = Property(item, "index");
        var score = Property(item, "relevance_score");
        if (index.ValueKind != JsonValueKind.Number || !index.TryGetInt32(out var i) || i < 0
            || score.ValueKind != JsonValueKind.Number || !score.TryGetSingle(out var value) || !float.IsFinite(value))
            throw new JsonException("Cirrascale returned an invalid ranking.");
        return new() { Index = i, RelevanceScore = value };
    }
}
