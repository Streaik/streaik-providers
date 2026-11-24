using System.Runtime.CompilerServices;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;

namespace Streaik.Providers;

public static class MixedbreadProviderRegistration
{
    public static IServiceCollection AddMixedbread(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, MixedbreadProvider.Id, "https://api.mixedbread.com/v1/");
        return services.AddProvider<MixedbreadProvider>(MixedbreadProvider.Id);
    }
}

internal sealed class MixedbreadProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), IResponsesProvider, INativeChatOptionsProvider
{
    internal const string Id = "mixedbread";

    public override IAsyncEnumerable<Model> ListModelsAsync(CancellationToken ct)
        => StaticModelListing.ListAsync(Id, ConfiguredApiKey(), ct);

    public async Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        return await NativeHttp.PostAsync<ResponseRequest, ResponseResult>(client, "responses", request, Bearer(ApiKey()), headers, ResponseJson.Default, ct);
    }

    public async IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        await foreach (var part in NativeHttp.StreamAsync<ResponseRequest, ResponseStreamPart>(client, "responses", request, Bearer(ApiKey()), headers,
            ResponseJson.Default, part => part.Type is "response.completed" or "response.failed" or "response.incomplete" or "error", ct))
            yield return part;
    }
}
