using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class NscaleProviderRegistration
{
    public static IServiceCollection AddNscale(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, NscaleProvider.Id, "https://inference.api.nscale.com/");
        return services.AddProvider<NscaleProvider>(NscaleProvider.Id);
    }
}

internal sealed class NscaleProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id, "v1/"), INativeChatOptionsProvider, IEmbeddingMetadataProvider, IImageModelRoutingProvider
{
    internal const string Id = "nscale";

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (ConfiguredApiKey() is not { } key || string.IsNullOrWhiteSpace(key)) yield break;
        var result = await FetchModelsAsync(key, ct);
        foreach (var item in Array(result.ValueKind == JsonValueKind.Array ? result : Property(result, "data")))
        {
            if (String(item, "id") is not { Length: > 0 } model || Property(item, "active").ValueKind == JsonValueKind.False) continue;
            var context = Property(item, "context_length");
            yield return new()
            {
                Id = model, Name = String(item, "name") ?? model, OwnedBy = String(item, "owned_by") ?? Id,
                Type = IsImageModel(model) ? "image" : model.Contains("embedding", StringComparison.OrdinalIgnoreCase) ? "embedding" : "language",
                Description = String(item, "description"),
                ContextWindow = context.ValueKind == JsonValueKind.Number && context.TryGetInt32(out var size) ? size : null,
                Pricing = ProviderPrices.ReadPair(Property(item, "pricing"), "input", "output")
            };
        }
    }

    private static bool IsImageModel(string model) => model.Contains("bytedance", StringComparison.OrdinalIgnoreCase)
        || model.Contains("stabilityai", StringComparison.OrdinalIgnoreCase) || model.Contains("black-forest-labs", StringComparison.OrdinalIgnoreCase);

    public Task<bool> IsImageModelAsync(string model, CancellationToken ct) => Task.FromResult(IsImageModel(model));

    public async Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        // The gateway emits its existing completed event for buffered generation when stream=true.
        return await CompatibleImageGeneration.GenerateAsync(client, Id, ApiKey(), request, headers, ct);
    }

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct) => throw new ProviderRequestException("Nscale does not support image edits.");

    public async Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct) => (await EmbedWithMetadataAsync(request, headers, ct)).Result;

    public async Task<(OpenAIEmbeddingResponse Result, JsonElement Body, Dictionary<string, string?> Headers)> EmbedWithMetadataAsync(
        OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, "v1/embeddings", request, Bearer(ApiKey()), headers, Json, ct);
        return (body.Deserialize<OpenAIEmbeddingResponse>(Json) ?? throw new JsonException("Missing embeddings."), body,
            NativeHttp.SafeResponseMetadataHeaders(upstream));
    }
}
