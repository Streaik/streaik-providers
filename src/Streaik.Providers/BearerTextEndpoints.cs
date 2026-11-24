using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Streaik.Messages;
using Streaik.Responses;
using Streaik.Responses.Streaming;

namespace Streaik.Providers;

internal class BearerTextEndpoints(IHttpClientFactory clients, IProviderCredentials credentials, string id)
    : IResponsesProvider, IMessagesProvider, INativeChatOptionsProvider
{
    public Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PostAsync<ResponseRequest, ResponseResult>("responses", request, headers, ResponseJson.Default, ct);

    public IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamAsync<ResponseRequest, ResponseStreamPart>("responses", request, headers, ResponseJson.Default,
            part => part.Type is "response.completed" or "response.failed" or "response.incomplete" or "error", ct);

    public Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PostAsync<MessagesRequest, MessagesResponse>("messages", request, headers, MessagesJson.Default, ct);

    public IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamAsync<MessagesRequest, MessageStreamPart>("messages", request, headers, MessagesJson.Default,
            part => part.Type is "message_stop" or "error", ct);

    private async Task<TResult> PostAsync<TRequest, TResult>(string path, TRequest request,
        IReadOnlyDictionary<string, string> headers, JsonSerializerOptions json, CancellationToken ct)
    {
        using var client = clients.CreateClient(id);
        return await NativeHttp.PostAsync<Dictionary<string, JsonElement>, TResult>(client, path,
            NativeOptionsPayload.Build(request, id, json, false), Auth(), headers, json, ct);
    }

    private async IAsyncEnumerable<TEvent> StreamAsync<TRequest, TEvent>(string path, TRequest request,
        IReadOnlyDictionary<string, string> headers, JsonSerializerOptions json, Func<TEvent, bool> terminal, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = clients.CreateClient(id);
        await foreach (var part in NativeHttp.StreamAsync<Dictionary<string, JsonElement>, TEvent>(client, path,
            NativeOptionsPayload.Build(request, id, json, true), Auth(), headers, json, terminal, ct)) yield return part;
    }

    private Action<HttpRequestHeaders> Auth()
    {
        var key = MediaHttp.RequiredKey(credentials, id);
        return headers => headers.Authorization = new("Bearer", key);
    }

    protected string? ConfiguredApiKey() => credentials.GetApiKey(id);
}
