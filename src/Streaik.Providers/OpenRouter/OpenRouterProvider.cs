using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.Messages;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.OpenRouter;

public static class OpenRouterRegistration
{
    public static IServiceCollection AddOpenRouter(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, OpenRouterProvider.Id, "https://openrouter.ai/api/");
        return services.AddProvider<OpenRouterProvider>(OpenRouterProvider.Id, "anthropic-version", "anthropic-beta");
    }
}

public sealed partial class OpenRouterProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IChatCompletionsProvider, IResponsesProvider, IMessagesProvider, IModelListingProvider, IEmbeddingsProvider, IRerankingProvider, IDecisionsProvider,
      IImageProvider, IStreamingImagesProvider, ISpeechProvider, ITranscriptionProvider, IVideoProvider
{
    public const string Id = "openrouter";

    public Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PostAsync<ChatCompletionOptions, ChatCompletion>("v1/chat/completions", request, headers, ct);

    public IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamAsync<ChatCompletionOptions, ChatCompletionUpdate>("v1/chat/completions", request, headers, Json, _ => false, ct);

    public Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PostAsync<ResponseRequest, ResponseResult>("v1/responses", request, headers, ct, ResponseJson.Default);

    public IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamAsync<ResponseRequest, ResponseStreamPart>("v1/responses", request, headers, ResponseJson.Default,
            part => part.Type is "response.completed" or "response.incomplete" or "response.failed" or "error", ct);

    public Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PostAsync<MessagesRequest, MessagesResponse>("v1/messages", request, headers, ct, MessagesJson.Default);

    public IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamAsync<MessagesRequest, MessageStreamPart>("v1/messages", request, headers, MessagesJson.Default,
            part => part.Type is "message_stop" or "error", ct);

    public Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PostAsync<OpenAIEmbeddingRequest, OpenAIEmbeddingResponse>("v1/embeddings", request, headers, ct);

    public async Task<RerankingResponse> RerankAsync(RerankingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Documents.Type != "text" || Array(request.Documents.Values).Any(x => x.ValueKind != JsonValueKind.String))
            throw new ProviderRequestException("OpenRouter reranking requires a text document array.", 422);
        var payload = new JsonObject();
        if (request.ProviderOptions?.TryGetValue(Id, out var options) == true && options.ValueKind == JsonValueKind.Object)
            foreach (var item in options.EnumerateObject()) payload[item.Name] = Copy(item.Value);
        payload["model"] = request.Model;
        payload["query"] = request.Query;
        payload["documents"] = Copy(request.Documents.Values);
        payload["top_n"] = request.TopN;
        var result = await PostAsync<JsonObject, JsonElement>("v1/rerank", payload, headers, ct);
        var ranking = Property(result, "results");
        if (ranking.ValueKind != JsonValueKind.Array) throw new JsonException("Missing reranking results.");
        var scores = new List<RerankingRanking>();
        foreach (var item in Array(ranking))
        {
            var i = Property(item, "index"); var score = Property(item, "relevance_score");
            if (i.ValueKind != JsonValueKind.Number || score.ValueKind != JsonValueKind.Number) throw new JsonException();
            scores.Add(new() { Index = i.GetInt32(), RelevanceScore = score.GetDouble() });
        }
        return new() { Ranking = scores, Response = new() { Id = String(result, "id"), Timestamp = DateTime.UtcNow }, ProviderMetadata = new() { [Id] = result } };
    }

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var key = credentials.GetApiKey(Id);
        if (string.IsNullOrWhiteSpace(key)) yield break;
        using var client = clients.CreateClient(Id);
        var seen = new HashSet<(string Id, string Type)>();
        foreach (var modalities in new[] { "all", "speech,transcription" })
        {
            JsonElement root;
            try { root = await NativeHttp.GetAsync<JsonElement>(client, $"v1/models?output_modalities={modalities}", headers => headers.Authorization = new("Bearer", key), ct); }
            catch (Exception error) when (modalities != "all" && !ct.IsCancellationRequested && error is HttpRequestException or JsonException or OperationCanceledException)
            { continue; }
            foreach (var item in Array(Property(root, "data")))
            {
                if (String(item, "id") is not { Length: > 0 } id) continue;
                var context = Property(item, "context_length");
                var limit = Property(Property(item, "top_provider"), "max_completion_tokens");
                foreach (var type in ModelTypes(item))
                    if (seen.Add((id, type)))
                        yield return new Model { Id = id, Name = String(item, "name") ?? id, Type = type, Description = String(item, "description"),
                            Created = ProviderModelJson.ReadCreated(item),
                            ContextWindow = context.ValueKind == JsonValueKind.Number && context.TryGetInt32(out var count) ? count : null,
                            MaxTokens = limit.ValueKind == JsonValueKind.Number && limit.TryGetInt32(out var max) ? max : null };
            }
        }
    }

    private static IEnumerable<string> ModelTypes(JsonElement item)
    {
        if (String(item, "type")?.Trim() is { Length: > 0 } explicitType && !explicitType.Equals("model", StringComparison.OrdinalIgnoreCase))
            return [explicitType];
        var architecture = Property(item, "architecture");
        static IEnumerable<string?> Modalities(JsonElement value) =>
            Array(value).Where(part => part.ValueKind == JsonValueKind.String).Select(part => part.GetString());
        var inputs = Modalities(Property(architecture, "input_modalities")).ToHashSet();
        var outputs = Modalities(Property(architecture, "output_modalities")).ToHashSet();
        if (outputs.Count == 0) return ["language"];
        // Models can appear in more than one modality, without losing text-chat eligibility.
        var audioOnlyInput = !inputs.Contains("text") && inputs.Overlaps(["audio", "speech"]);
        var types = new HashSet<string>(StringComparer.Ordinal);
        foreach (var output in outputs)
        {
            if ((output switch
                {
                    "text" => audioOnlyInput ? "transcription" : "language",
                    "transcription" or "image" or "video" or "speech" => output,
                    "audio" when !outputs.Contains("text") && inputs.Contains("text") => "speech",
                    "audio" => "audio",
                    "embeddings" => "embedding",
                    "rerank" => "reranking",
                    "decisions" => "decision",
                    _ => null,
                }) is { } type) types.Add(type);
        }
        return types.Count > 0 ? types : ["other"];
    }

    private Action<HttpRequestHeaders> Auth()
    {
        var key = RequiredKey(credentials, Id);
        return headers => headers.Authorization = new("Bearer", key);
    }

    private async Task<TResult> PostAsync<TRequest, TResult>(string path, TRequest request, IReadOnlyDictionary<string, string> headers,
        CancellationToken ct, JsonSerializerOptions? json = null)
    {
        var auth = Auth();
        using var client = clients.CreateClient(Id);
        return await NativeHttp.PostAsync<TRequest, TResult>(client, path, request, auth, headers, json ?? Json, ct);
    }

    private async IAsyncEnumerable<TEvent> StreamAsync<TRequest, TEvent>(string path, TRequest request, IReadOnlyDictionary<string, string> headers,
        JsonSerializerOptions json, Func<TEvent, bool> terminal, [EnumeratorCancellation] CancellationToken ct)
    {
        var auth = Auth();
        using var client = clients.CreateClient(Id);
        await foreach (var part in NativeHttp.StreamAsync(client, path, request, auth, headers, json, terminal, ct)) yield return part;
    }
}
