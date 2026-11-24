using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;

namespace Streaik.Providers.OpenAI;

public static class OpenAIRegistration
{
    public static IServiceCollection AddOpenAI(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, OpenAIProvider.Id, "https://api.openai.com/");
        return services.AddProvider<OpenAIProvider>(OpenAIProvider.Id, "X-Client-Request-Id");
    }
}

public sealed partial class OpenAIProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IChatCompletionsProvider, IResponsesProvider, IModelListingProvider, IImageProvider, ISpeechProvider, ITranscriptionProvider,
      IEmbeddingsProvider, IRealtimeTokenProvider, IRealtimeSessionProvider, IContainerFilesProvider, IStreamingImagesProvider,
      IOpenAIRealtimeProvider, IOpenAILiveSessionProvider, INativeAudioTranscriptionProvider, IStreamingTranscriptionProvider, IDecisionsProvider,
      ISkillReadProvider, ISkillUploadProvider
{
    public const string Id = "openai";

    public Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PostAsync<ChatCompletionOptions, ChatCompletion>("v1/chat/completions", request, headers, JsonSerializerOptions.Web, ct);

    public IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamAsync<ChatCompletionOptions, ChatCompletionUpdate>("v1/chat/completions", request, headers, JsonSerializerOptions.Web, _ => false, ct);

    public Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PostAsync<ResponseRequest, ResponseResult>("v1/responses", request, headers, ResponseJson.Default, ct);

    public IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamAsync<ResponseRequest, ResponseStreamPart>("v1/responses", request, headers, ResponseJson.Default,
            part => part.Type is "response.completed" or "response.failed" or "response.incomplete" or "error", ct);

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var key = credentials.GetApiKey(Id);
        if (string.IsNullOrWhiteSpace(key)) yield break;
        using var client = clients.CreateClient(Id);
        var page = await NativeHttp.GetAsync<ModelResponse>(client, "v1/models", h => Authenticate(h, key), ct, ProviderModelJson.Options);
        foreach (var model in page.Data) yield return model;
    }

    private async Task<TResult> PostAsync<TRequest, TResult>(string path, TRequest request, IReadOnlyDictionary<string, string> headers,
        JsonSerializerOptions json, CancellationToken ct)
    {
        var key = RequireKey();
        using var client = clients.CreateClient(Id);
        return await NativeHttp.PostAsync<TRequest, TResult>(client, path, request, h => Authenticate(h, key), headers, json, ct);
    }

    private async IAsyncEnumerable<TEvent> StreamAsync<TRequest, TEvent>(string path, TRequest request,
        IReadOnlyDictionary<string, string> headers, JsonSerializerOptions json, Func<TEvent, bool> terminal, [EnumeratorCancellation] CancellationToken ct)
    {
        var key = RequireKey();
        using var client = clients.CreateClient(Id);
        await foreach (var part in NativeHttp.StreamAsync(client, path, request, h => Authenticate(h, key), headers, json, terminal, ct))
            yield return part;
    }

    private string RequireKey() => credentials.GetApiKey(Id) is { } key && !string.IsNullOrWhiteSpace(key)
        ? key : throw new ProviderCredentialException(Id);

    private static void Authenticate(HttpRequestHeaders headers, string key)
        => headers.Authorization = new AuthenticationHeaderValue("Bearer", key);
}
