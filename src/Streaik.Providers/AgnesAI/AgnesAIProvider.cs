using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class AgnesAIProviderRegistration
{
    public static IServiceCollection AddAgnesAI(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AgnesAIProvider.Id, "https://apihub.agnes-ai.com/");
        services.AddHttpClient(AgnesAIProvider.DeliveryClient).ConfigurePrimaryHttpMessageHandler(AgnesAIProvider.CreateDeliveryHandler);
        return services.AddProvider<AgnesAIProvider>(AgnesAIProvider.Id);
    }
}

internal sealed partial class AgnesAIProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id, "v1/"), INativeChatOptionsProvider, IResponsesProvider,
        IImageModelRoutingProvider, IStreamingImagesProvider, IVideoProvider
{
    internal const string Id = "agnesai";
    internal const string DeliveryClient = Id + "-delivery";
    private readonly BearerTextEndpoints text = new(clients, credentials, Id, prefix: "v1/");

    public Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.CreateResponseAsync(request, headers, ct);

    public IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.StreamResponseAsync(request, headers, ct);

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (ConfiguredApiKey() is not { } key || string.IsNullOrWhiteSpace(key)) yield break;
        var result = await FetchModelsAsync(key, ct);
        foreach (var item in Array(Property(result, "data")))
        {
            if (String(item, "id") is not { Length: > 0 } id) continue;
            yield return new() { Id = id, Name = id, OwnedBy = String(item, "owned_by") ?? Id, Type = ModelType(id) };
        }
    }

    public Task<bool> IsImageModelAsync(string model, CancellationToken ct) => Task.FromResult(ModelType(model) == "image");

    private static string ModelType(string model) => model.Contains("video", StringComparison.OrdinalIgnoreCase) ? "video"
        : model.Contains("image", StringComparison.OrdinalIgnoreCase) ? "image" : "language";
}
