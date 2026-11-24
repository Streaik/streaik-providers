using System.Runtime.CompilerServices;
using Microsoft.Extensions.DependencyInjection;
using Streaik.Responses;
using Streaik.Responses.Streaming;

namespace Streaik.Providers;

public static class CencoriRegistration
{
    public static IServiceCollection AddCencori(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, "cencori", "https://api.cencori.com/v1/");
        return services.AddProvider<CencoriProvider>("cencori");
    }
}

internal sealed class CencoriProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, "cencori"), IResponsesProvider, INativeChatOptionsProvider
{
    public async Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var auth = Bearer(ApiKey());
        using var client = CreateClient();
        return await NativeHttp.PostAsync<ResponseRequest, ResponseResult>(client, "responses", request, auth, headers, ResponseJson.Default, ct);
    }

    public async IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        var auth = Bearer(ApiKey());
        using var client = CreateClient();
        await foreach (var part in NativeHttp.StreamAsync<ResponseRequest, ResponseStreamPart>(client, "responses", request, auth, headers,
            ResponseJson.Default, part => part.Type is "response.completed" or "response.failed" or "response.incomplete" or "error", ct))
            yield return part;
    }
}
