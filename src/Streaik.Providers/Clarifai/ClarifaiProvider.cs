using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class ClarifaiProviderRegistration
{
    public static IServiceCollection AddClarifai(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, ClarifaiProvider.Id, "https://api.clarifai.com/v2/ext/openai/v1/");
        ProviderServiceCollectionExtensions.AddClient(services, ClarifaiProvider.CatalogClient, "https://api.clarifai.com/");
        return services.AddProvider<ClarifaiProvider>(ClarifaiProvider.Id);
    }
}

internal sealed class ClarifaiProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatEmbeddingProvider(clients, credentials, Id, ""), IResponsesProvider, IImageProvider
{
    internal const string Id = "clarifai";
    internal const string CatalogClient = "clarifai-catalog";
    private readonly BearerTextEndpoints text = new(clients, credentials, Id);

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (ConfiguredApiKey() is not { } key || string.IsNullOrWhiteSpace(key)) yield break;
        using var client = CreateClient(CatalogClient);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        string? previousPage = null;
        for (var page = 1; ; page++)
        {
            var catalog = await NativeHttp.GetAsync<JsonElement>(client, $"v2/models?page={page}", h => h.Authorization = new("Key", key), ct);
            var models = Property(catalog, "models");
            if (models.ValueKind != JsonValueKind.Array || models.GetArrayLength() == 0) yield break;
            var raw = models.GetRawText();
            if (StringComparer.Ordinal.Equals(raw, previousPage)) yield break;
            previousPage = raw;
            foreach (var item in models.EnumerateArray())
            {
                var owner = String(item, "user_id");
                var app = String(item, "app_id");
                var name = String(item, "name");
                var type = String(item, "model_type_id");
                if (string.IsNullOrWhiteSpace(owner) || string.IsNullOrWhiteSpace(app) || string.IsNullOrWhiteSpace(name)
                    || string.IsNullOrWhiteSpace(type) || !HasOpenAITransport(item)) continue;
                var id = $"https://clarifai.com/{Uri.EscapeDataString(owner)}/{Uri.EscapeDataString(app)}/models/{Uri.EscapeDataString(name)}";
                if (seen.Add(id)) yield return new Model { Id = id, Name = name, OwnedBy = owner, Description = String(item, "description"),
                    Type = type.ToLowerInvariant() switch { "text-to-text" => "language", "text-to-image" => "image", "text-embedder" => "embedding", _ => type } };
            }
        }
    }

    private static bool HasOpenAITransport(JsonElement model)
    {
        var methods = Array(Property(Property(model, "model_version"), "method_signatures")).Select(method => String(method, "name")).ToHashSet();
        return methods.Contains("openai_transport") && methods.Contains("openai_stream_transport");
    }

    public Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.CreateResponseAsync(request, headers, ct);

    public IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.StreamResponseAsync(request, headers, ct);

    public async Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Stream == true) request.ResponseFormat = "b64_json";
        using var client = CreateClient();
        var result = await CompatibleImageGeneration.GenerateAsync(client, Id, ApiKey(), request, headers, ct, "images/generations");
        result.Size ??= request.Size;
        result.Quality ??= request.Quality;
        result.Background ??= request.Background;
        return result;
    }

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => throw new ProviderRequestException("Clarifai does not support image editing.");
}
