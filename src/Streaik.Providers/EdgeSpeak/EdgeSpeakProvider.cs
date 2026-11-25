using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Streaik.Messages;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class EdgeSpeakRegistration
{
    public static IServiceCollection AddEdgeSpeak(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, EdgeSpeakProvider.Id, "http://127.0.0.1:1118/v1/");
        services.AddHttpClient(EdgeSpeakProvider.Id).ConfigureHttpClient((sp, client) =>
        {
            var endpoint = sp.GetService<IConfiguration>()?["Providers:edgespeak:Endpoint"];
            if (string.IsNullOrWhiteSpace(endpoint)) return;
            if (!Uri.TryCreate(endpoint, UriKind.Absolute, out var uri) || uri.Scheme is not ("http" or "https") ||
                string.IsNullOrEmpty(uri.Host) || uri.UserInfo.Length != 0 || uri.Query.Length != 0 || uri.Fragment.Length != 0)
                throw new InvalidOperationException("EdgeSpeak endpoint must be an absolute HTTP(S) base URL without credentials, query, or fragment.");
            var root = uri.AbsoluteUri.TrimEnd('/');
            client.BaseAddress = new Uri(root.EndsWith("/v1", StringComparison.Ordinal) ? root + "/" : root + "/v1/");
        });
        return services.AddProvider<EdgeSpeakProvider>(EdgeSpeakProvider.Id);
    }
}

internal sealed partial class EdgeSpeakProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, "edgespeak"), IResponsesProvider, IMessagesProvider, INativeChatOptionsProvider,
        ISpeechProvider, IStreamingSpeechEventsProvider, ITranscriptionProvider, IStreamingTranscriptionProvider, INativeAudioTranscriptionProvider
{
    internal const string Id = "edgespeak";
    private readonly BearerTextEndpoints _text = new(clients, credentials, Id, allowAnonymous: true);

    protected override Action<HttpRequestHeaders> ChatAuth()
        => ConfiguredApiKey() is { } key && !string.IsNullOrWhiteSpace(key) ? Bearer(key) : _ => { };

    public Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => _text.CreateResponseAsync(request, headers, ct);

    public IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => _text.StreamResponseAsync(request, headers, ct);

    public Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => _text.CreateMessageAsync(request, MessageHeaders(headers), ct);

    public IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => _text.StreamMessageAsync(request, MessageHeaders(headers), ct);

    private static IReadOnlyDictionary<string, string> MessageHeaders(IReadOnlyDictionary<string, string> headers)
    {
        var result = new Dictionary<string, string>(headers, StringComparer.OrdinalIgnoreCase);
        result.TryAdd("anthropic-version", "2023-06-01");
        return result;
    }

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        var body = await NativeHttp.GetAsync<JsonElement>(client, "models", ChatAuth(), ct);
        foreach (var item in Array(Property(body, "data")))
        {
            var endpoints = Array(Property(item, "supported_endpoints")).Where(p => p.ValueKind == JsonValueKind.String)
                .Select(p => p.GetString()).ToArray();
            var type = endpoints.Any(p => p is "/v1/chat/completions" or "/v1/responses" or "/v1/messages") ? "language"
                : endpoints.Contains("/v1/audio/speech") ? "speech" : endpoints.Contains("/v1/audio/transcriptions") ? "transcription" : null;
            if (type is null || String(item, "id") is not { Length: > 0 } model) continue;
            yield return new() { Id = model, Name = String(item, "name") ?? model, OwnedBy = String(item, "owned_by") ?? "EdgeSpeak",
                Type = type, Description = String(item, "description"),
                Created = Property(item, "created") is { ValueKind: JsonValueKind.Number } created && created.TryGetInt64(out var seconds) ? seconds : null };
        }
    }

    private async Task<HttpResponseMessage> SendMediaAsync(string path, HttpContent content, string accept,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct, bool bufferedSpeech = false)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, path) { Content = content };
        ChatAuth()(request.Headers);
        request.Headers.Accept.ParseAdd(accept);
        if (bufferedSpeech) request.Headers.Add("buffered-speech", "1");
        NativeHttp.ApplyHeaders(request, headers);
        using var client = CreateClient();
        return await NativeHttp.SendAsync(client, request, ct);
    }

    private static Dictionary<string, JsonElement> ScopedOptions(IReadOnlyDictionary<string, JsonElement>? options)
        => options?.TryGetValue(Id, out var scoped) == true ? Fields(scoped) : new(StringComparer.Ordinal);

    private static Dictionary<string, JsonElement> Fields(JsonElement options)
        => options.ValueKind == JsonValueKind.Object ? options.EnumerateObject().ToDictionary(p => p.Name, p => p.Value, StringComparer.Ordinal)
            : throw new ProviderRequestException("EdgeSpeak provider options must be an object.");

    private static object RawChunk(string type, JsonElement data) => new { @event = type, data };
    private static Dictionary<string, JsonElement> Metadata(JsonElement data) => new() { [Id] = data };
    private static JsonElement Value<T>(T value) => JsonSerializer.SerializeToElement(value, Json);
    private static string RequiredString(JsonElement value, string name)
        => String(value, name) ?? throw new JsonException($"EdgeSpeak response is missing '{name}'.");
    private static double? Number(JsonElement value, string name)
        => Property(value, name) is { ValueKind: JsonValueKind.Number } number && number.TryGetDouble(out var n) && double.IsFinite(n) ? n : null;

    private static Dictionary<string, string> ResponseHeaders(HttpResponseMessage response)
        => NativeHttp.SafeResponseMetadataHeaders(response.Headers.Concat(response.Content.Headers)
            .GroupBy(p => p.Key, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(g => g.Key, g => (string?)string.Join(", ", g.SelectMany(p => p.Value)), StringComparer.OrdinalIgnoreCase))
            .ToDictionary(p => p.Key, p => p.Value!, StringComparer.OrdinalIgnoreCase);

    private static async IAsyncEnumerable<(string Type, JsonElement Data)> Events(HttpResponseMessage response, string terminal, [EnumeratorCancellation] CancellationToken ct)
    {
        if (response.Content.Headers.ContentType?.MediaType != "text/event-stream")
            throw new IOException("EdgeSpeak did not return an event stream.");
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        await foreach (var item in System.Net.ServerSentEvents.SseParser.Create(stream).EnumerateAsync(ct))
        {
            if (item.Data == "[DONE]") break;
            using var doc = JsonDocument.Parse(item.Data);
            var body = doc.RootElement.Clone();
            var payloadType = String(body, "type");
            var type = item.EventType is "message" or "" ? payloadType ?? "" : item.EventType;
            if (type == "error" || payloadType == "error" || Property(body, "error").ValueKind is not (JsonValueKind.Null or JsonValueKind.Undefined))
                throw new IOException("EdgeSpeak returned an error event.");
            if (payloadType is not null && payloadType != type) throw new JsonException("Conflicting EdgeSpeak event types.");
            yield return (type, body);
            if (type == terminal) yield break;
        }
        throw new IOException($"EdgeSpeak stream disconnected before '{terminal}'.");
    }
}
