using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Streaik.Core.Models;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class CohereProvider : IEmbeddingMetadataProvider, IRerankingProvider, ITranscriptionProvider,
    INativeAudioTranscriptionProvider, IChatAudioRoutingProvider
{
    public bool IsSpeechModel(string model) => false;
    bool IChatAudioRoutingProvider.IsTranscriptionModel(string model) => IsTranscriptionModel(model);
    private static bool IsTranscriptionModel(string model) => model.Contains("transcrib", StringComparison.OrdinalIgnoreCase)
        || model.Contains("transcription", StringComparison.OrdinalIgnoreCase) || model.Contains("speech-to-text", StringComparison.OrdinalIgnoreCase);

    public async Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => (await EmbedWithMetadataAsync(request, headers, ct)).Result;

    public async Task<(OpenAIEmbeddingResponse Result, JsonElement Body, Dictionary<string, string?> Headers)> EmbedWithMetadataAsync(
        OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var encoding = request.EncodingFormat?.ToLowerInvariant() ?? "float";
        if (encoding is not ("float" or "base64")) throw new ProviderRequestException("Cohere embedding encoding must be float or base64.");
        if (request.Dimensions is <= 0) throw new ProviderRequestException("Embedding dimensions must be positive.");
        var payload = CopyOptions(request.AdditionalProperties, "model", "input", "texts", "dimensions", "encoding_format", "embedding_types", "user");
        payload["model"] = request.Model;
        payload["input_type"] ??= JsonValue.Create("search_document");
        if (payload["input_type"] is not JsonValue inputType || !inputType.TryGetValue<string>(out var type)
            || type is not ("search_document" or "search_query" or "classification" or "clustering" or "image"))
            throw new ProviderRequestException("Invalid Cohere embedding input_type.");
        var images = payload["images"];
        var inputs = payload["inputs"];
        if ((images is not null && images is not JsonArray { Count: > 0 }) || (inputs is not null && inputs is not JsonArray { Count: > 0 })
            || (images is not null && inputs is not null) || (type == "image" && images is null))
            throw new ProviderRequestException("Cohere embeddings accepts one native images or inputs array; input_type 'image' requires images.");
        if (images is null && inputs is null)
        {
            if (request.Input.ValueKind == JsonValueKind.String) payload["texts"] = new JsonArray(JsonValue.Create(request.Input.GetString()));
            else if (request.Input.ValueKind == JsonValueKind.Array && request.Input.GetArrayLength() > 0
                && Array(request.Input).All(x => x.ValueKind == JsonValueKind.String)) payload["texts"] = Copy(request.Input);
            else throw new ProviderRequestException("Cohere embeddings requires text strings or a native inputs/images array.");
        }
        if (request.Dimensions is { } dimension) payload["output_dimension"] = dimension;
        payload["embedding_types"] = new JsonArray(JsonValue.Create(encoding));
        using var client = Client();
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, "v2/embed", payload, Auth(), headers, Json, ct);
        var vectors = Property(Property(body, "embeddings"), encoding);
        if (vectors.ValueKind != JsonValueKind.Array) throw new JsonException("Cohere returned no embedding vectors.");
        var data = Array(vectors).Select((vector, i) =>
        {
            if (encoding == "float" && vector.ValueKind != JsonValueKind.Array || encoding == "base64" && vector.ValueKind != JsonValueKind.String)
                throw new JsonException("Cohere returned an invalid embedding vector.");
            return new OpenAIEmbeddingData { Index = i, Embedding = vector };
        }).ToArray();
        var expected = (images as JsonArray ?? inputs as JsonArray ?? (JsonArray)payload["texts"]!).Count;
        if (data.Length != expected) throw new JsonException("Cohere returned an unexpected number of embeddings.");
        var meta = Property(body, "meta");
        var count = 0;
        foreach (var section in new[] { "billed_units", "tokens" })
            if (Int(Property(meta, section), "input_tokens") is { } tokens) { count = tokens; break; }
        return (new() { Model = request.Model, Data = data, Usage = new() { PromptTokens = count, TotalTokens = count } }, body, upstream);
    }

    public async Task<RerankingResponse> RerankAsync(RerankingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var docs = request.Documents?.Values ?? default;
        if (docs.ValueKind != JsonValueKind.Array || docs.GetArrayLength() == 0) throw new ProviderRequestException("Cohere requires documents to rerank.");
        var body = CopyOptions(Options(request.ProviderOptions), "model", "query", "documents", "top_n");
        body["model"] = request.Model;
        body["query"] = request.Query;
        body["documents"] = Copy(docs);
        if (request.TopN is { } top) body["top_n"] = top;
        using var client = Client();
        var (raw, upstream) = await NativeHttp.PostJsonAsync(client, "v2/rerank", body, Auth(), headers, Json, ct);
        var results = Property(raw, "results");
        if (results.ValueKind != JsonValueKind.Array) throw new JsonException("Cohere returned no reranking results.");
        var ranking = Array(results).Select(row => new RerankingRanking
        {
            Index = Int(row, "index") is >= 0 and var index && index < docs.GetArrayLength() ? index : throw new JsonException("Invalid reranking index."),
            RelevanceScore = Property(row, "relevance_score") is { ValueKind: JsonValueKind.Number } s && s.TryGetDouble(out var score) && double.IsFinite(score)
                ? score : throw new JsonException("Invalid relevance score.")
        }).ToArray();
        return new() { Ranking = ranking, ProviderMetadata = Scoped(raw), Response = new() { Id = String(raw, "id"), ModelId = request.Model,
            Timestamp = DateTime.UtcNow, Body = raw, Headers = HeaderValues(upstream) } };
    }

    private static IEnumerable<KeyValuePair<string, JsonElement>> Options(Dictionary<string, JsonElement>? options)
    {
        if (options?.TryGetValue(Id, out var provider) != true || provider.ValueKind == JsonValueKind.Null) yield break;
        if (provider.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Cohere provider options must be an object.");
        foreach (var p in provider.EnumerateObject()) yield return KeyValuePair.Create(p.Name, p.Value);
    }

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName, string? language,
        IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(language) && options?.TryGetValue("language", out var lang) == true && lang.ValueKind == JsonValueKind.String)
            language = lang.GetString();
        if (string.IsNullOrWhiteSpace(language)) throw new ProviderRequestException("Cohere transcription requires a language.");
        using var client = Client();
        using var form = new MultipartFormDataContent();
        var fields = CopyOptions(options, "model", "file", "language", "response_format", "stream");
        fields["model"] = model;
        fields["language"] = language;
        AddFields(form, fields);
        AddFile(form, "file", audio, mediaType, fileName);
        using var request = new HttpRequestMessage(HttpMethod.Post, "v2/audio/transcriptions") { Content = form };
        Auth()(request.Headers);
        NativeHttp.ApplyHeaders(request, headers);
        using var response = await NativeHttp.SendAsync(client, request, ct);
        var raw = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        var text = String(raw, "text") ?? throw new JsonException("Cohere returned no transcription text.");
        return new() { Text = text, Language = String(raw, "language") ?? language, ProviderMetadata = Scoped(raw),
            Response = new() { ModelId = model, Timestamp = DateTime.UtcNow, Body = raw, Headers = HeaderValues(response.Headers.Concat(response.Content.Headers)
                .Select(h => KeyValuePair.Create<string, string?>(h.Key, string.Join(", ", h.Value)))) } };
    }

    public async Task<HttpResponseMessage> TranscribeNativeAsync(string model, IFormCollection request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var format = request["response_format"].ToString().Trim().ToLowerInvariant();
        if (format is not ("" or "json" or "text" or "verbose_json")) throw new ProviderRequestException("Unsupported Cohere transcription response format.");
        var file = request.Files.GetFile("file") ?? throw new ProviderRequestException("An audio file is required.");
        var options = request.Where(p => p.Key is not ("model" or "file" or "language" or "response_format" or "stream"))
            .ToDictionary(p => p.Key, p => JsonSerializer.SerializeToElement(p.Value.ToString()), StringComparer.Ordinal);
        await using var audio = file.OpenReadStream();
        var result = await TranscribeAsync(model, audio, file.ContentType, file.FileName, request["language"], options, headers, ct);
        if (request["stream"] == "true")
        {
            var sse = new StringBuilder();
            void Frame<T>(T evt) => sse.Append("data: ").Append(JsonSerializer.Serialize(evt, Json)).Append("\n\n");
            if (result.Text.Length != 0) Frame(new OpenAITranscriptionTextDelta { Delta = result.Text });
            Frame(new OpenAITranscriptionTextDone { Text = result.Text });
            return new() { Content = new StringContent(sse.ToString(), Encoding.UTF8, "text/event-stream") };
        }
        HttpContent content = format switch
        {
            "text" => new StringContent(result.Text, Encoding.UTF8, "text/plain"),
            "verbose_json" => JsonContent.Create(new OpenAITranscriptionVerboseResponse { Text = result.Text,
                Language = result.Language!, Duration = result.DurationInSeconds ?? 0, Segments = [], Words = [] }, options: Json),
            _ => JsonContent.Create(new OpenAITranscriptionResponse { Text = result.Text }, options: Json)
        };
        return new() { Content = content };
    }

    private static Dictionary<string, string> HeaderValues(IEnumerable<KeyValuePair<string, string?>> headers)
    {
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, value) in headers)
            if (value is not null && !name.Equals("Set-Cookie", StringComparison.OrdinalIgnoreCase)) result[name] = value;
        return result;
    }
}
