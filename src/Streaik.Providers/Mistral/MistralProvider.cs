using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Mistral;

public static class MistralProviderRegistration
{
    public static IServiceCollection AddMistral(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, MistralProvider.Id, "https://api.mistral.ai/v1/");
        return services.AddProvider<MistralProvider>(MistralProvider.Id);
    }
}

internal sealed partial class MistralProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), INativeChatOptionsProvider, IResponsesProvider, IEmbeddingsProvider,
      ISpeechProvider, IStreamingSpeechEventsProvider, ITranscriptionProvider, IStreamingTranscriptionProvider,
      INativeAudioTranscriptionProvider, IChatAudioRoutingProvider, IRealtimeTokenProvider
{
    internal const string Id = "mistral";

    public async Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        return await NativeHttp.PostAsync<OpenAIEmbeddingRequest, OpenAIEmbeddingResponse>(client, "embeddings", request,
            Bearer(ApiKey()), headers, Json, ct);
    }

    public async Task<ResponseResult> CreateResponseAsync(ResponseRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var body = ResponseBody(request, false);
        if (IsOcrModel(request.Model)) return OcrResponse(request.Model!, await ReadOcrAsync(request.Model!, ResponseOcrFiles(body), body, headers, ct));
        using var client = CreateClient();
        var raw = await NativeHttp.PostAsync<JsonObject, JsonElement>(client, "responses", body,
            Bearer(ApiKey()), headers, ResponseJson.Default, ct);
        return ReadResponse<ResponseResult>(raw, stream: false);
    }

    public async IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        var body = ResponseBody(request, true);
        if (IsOcrModel(request.Model))
        {
            var result = OcrResponse(request.Model!, await ReadOcrAsync(request.Model!, ResponseOcrFiles(body), body, headers, ct));
            foreach (var part in OcrResponseEvents(result)) yield return part;
            yield break;
        }

        using var client = CreateClient();
        var terminal = false;
        await foreach (var raw in NativeHttp.StreamAsync<JsonObject, JsonElement>(client, "responses", body, Bearer(ApiKey()), headers,
            ResponseJson.Default, _ => terminal, ct))
        {
            var part = ReadResponse<ResponseStreamPart>(Unwrap(raw, "type"), stream: true);
            terminal = part.Type is "response.completed" or "response.failed" or "response.incomplete" or "error";
            yield return part;
        }
        if (!terminal) throw new IOException("Mistral Responses ended before a terminal event.");
    }

    private static T ReadResponse<T>(JsonElement data, bool stream) where T : class
    {
        // Mistral sends completion times as decimal Unix seconds (e.g. 1795852935.0).
        // Normalize only that field to the Responses contract's whole seconds.
        var completed = Property(stream ? Property(data, "response") : data, "completed_at");
        if (completed.ValueKind == JsonValueKind.Number && !completed.TryGetInt64(out _)
            && completed.TryGetDecimal(out var seconds) && seconds is >= long.MinValue and <= long.MaxValue)
        {
            var root = JsonNode.Parse(data.GetRawText())!.AsObject();
            var response = stream ? root["response"]!.AsObject() : root;
            response["completed_at"] = (long)seconds;
            return root.Deserialize<T>(ResponseJson.Default) ?? throw new JsonException("Missing Mistral Responses payload.");
        }

        return data.Deserialize<T>(ResponseJson.Default) ?? throw new JsonException("Missing Mistral Responses payload.");
    }

    private static JsonObject ResponseBody(ResponseRequest request, bool stream)
    {
        if (request.Store == true || request.Background == true || request.PreviousResponseId is not null || request.Conversation is not null)
            throw new ProviderRequestException("Mistral Responses are stateless. Send the full history in input.");
        var body = JsonSerializer.SerializeToNode(NativeOptionsPayload.Build(request, Id, ResponseJson.Default, stream))!.AsObject();
        if (body["input"] is JsonArray input && input.OfType<JsonObject>().Any(item => NodeString(item["type"]) == "item_reference"))
            throw new ProviderRequestException("Mistral Responses require inline input items rather than item references.");
        if (body["tools"] is JsonArray tools && tools.Any(tool => tool is not JsonObject item || NodeString(item["type"]) is not ("function" or "namespace")))
            throw new ProviderRequestException("Mistral Responses only support client-executed function tools.", 422);
        // Known OpenAI-only transport fields are not Mistral API options. Keep the supported
        // request fields and native JSON extensions, including inline reasoning history.
        foreach (var name in new[] { "include", "stream_options", "context_management", "access_programs", "safety_identifier",
            "user", "prompt_cache_retention", "prompt_cache_options", "service_tier", "previous_response_id", "conversation" })
            body.Remove(name);
        if (body["reasoning"] is JsonObject reasoning)
        {
            reasoning.Remove("summary");
            if (reasoning.Count == 0) body.Remove("reasoning");
        }
        return body;
    }

    private static JsonElement Unwrap(JsonElement value, string discriminator)
        => Property(value, discriminator).ValueKind == JsonValueKind.Undefined && Property(value, "data") is { ValueKind: JsonValueKind.Object } data
            ? data : value;

    private static string? NodeString(JsonNode? node) => node is JsonValue value && value.TryGetValue<string>(out var text) ? text : null;
}
