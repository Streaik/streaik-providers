using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class NebiusProviderRegistration
{
    public static IServiceCollection AddNebius(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, NebiusProvider.Id, "https://api.tokenfactory.nebius.com/v1/");
        return services.AddProvider<NebiusProvider>(NebiusProvider.Id);
    }
}

internal sealed class NebiusProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), IResponsesProvider, INativeChatOptionsProvider, IEmbeddingMetadataProvider
{
    internal const string Id = "nebius";

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

    public async Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => (await EmbedWithMetadataAsync(request, headers, ct)).Result;

    public async Task<(OpenAIEmbeddingResponse Result, JsonElement Body, Dictionary<string, string?> Headers)> EmbedWithMetadataAsync(
        OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, "embeddings", request, Bearer(ApiKey()), headers, Json, ct);
        if (Property(body, "data") is not { ValueKind: JsonValueKind.Array } data || data.GetArrayLength() == 0)
            throw new JsonException("Nebius did not return embeddings.");
        var result = body.Deserialize<OpenAIEmbeddingResponse>(Json) ?? throw new JsonException("Missing Nebius embedding response.");
        result.Model = String(body, "model") ?? request.Model;
        return (result, body, NativeHttp.SafeResponseMetadataHeaders(upstream));
    }
}
