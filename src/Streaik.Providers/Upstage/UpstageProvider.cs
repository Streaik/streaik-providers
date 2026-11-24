using System.Runtime.CompilerServices;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;

namespace Streaik.Providers;

public static class UpstageProviderRegistration
{
    public static IServiceCollection AddUpstage(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, UpstageProvider.Id, "https://api.upstage.ai/");
        return services.AddProvider<UpstageProvider>(UpstageProvider.Id);
    }
}

internal sealed partial class UpstageProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id, "v1/"), IResponsesProvider, IDecisionsProvider, INativeChatOptionsProvider
{
    internal const string Id = "upstage";

    public override Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => base.CompleteChatAsync(PrepareChat(request), headers, ct);

    public override IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => base.StreamChatAsync(PrepareChat(request), headers, ct);

    private static ChatCompletionOptions PrepareChat(ChatCompletionOptions request)
    {
        request.Store = null;
        if (request.Tools?.Any() != true) { request.Tools = null!; request.ToolChoice = null; }
        return request;
    }

    public override IAsyncEnumerable<Model> ListModelsAsync(CancellationToken ct)
        => StaticModelListing.ListAsync(Id, ConfiguredApiKey(), ct);

    public async Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        return await NativeHttp.PostAsync<ResponseRequest, ResponseResult>(client, "v2/responses", request, Bearer(ApiKey()), headers, ResponseJson.Default, ct);
    }

    public async IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        await foreach (var part in NativeHttp.StreamAsync<ResponseRequest, ResponseStreamPart>(client, "v2/responses", request, Bearer(ApiKey()), headers,
            ResponseJson.Default, part => part.Type is "response.completed" or "response.failed" or "response.incomplete" or "error", ct))
            yield return part;
    }
}
