using System.Net;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed class BergetAIProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, "bergetai"), INativeChatOptionsProvider, IChatAudioRoutingProvider,
      ITranscriptionProvider, INativeAudioTranscriptionProvider, IRerankingProvider
{
    private const string TranscriptionsPath = "audio/transcriptions";

    public bool IsSpeechModel(string model) => false;
    public bool IsTranscriptionModel(string model) => model.Contains("whisper", StringComparison.OrdinalIgnoreCase)
        || model.Contains("transcription", StringComparison.OrdinalIgnoreCase)
        || model.Contains("speech-to-text", StringComparison.OrdinalIgnoreCase);

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (ConfiguredApiKey() is not { Length: > 0 } key || string.IsNullOrWhiteSpace(key)) yield break;
        var raw = await FetchModelsAsync(key, ct);
        foreach (var item in Array(Property(raw, "data")))
        {
            if (String(item, "id") is not { Length: > 0 } id || Property(item, "active").ValueKind == JsonValueKind.False) continue;
            var created = Property(item, "created");
            var context = Property(item, "context_length");
            yield return new()
            {
                Id = id, Name = String(item, "name") ?? id, OwnedBy = String(item, "owned_by") ?? ProviderId,
                Description = String(item, "description"),
                Type = String(item, "type") ?? (IsTranscriptionModel(id) ? "transcription"
                    : id.Contains("rerank", StringComparison.OrdinalIgnoreCase) ? "reranking" : "language"),
                Created = created.ValueKind == JsonValueKind.Number && created.TryGetInt64(out var ms) ? ms / 1000 : null,
                ContextWindow = context.ValueKind == JsonValueKind.Number && context.TryGetInt32(out var size) ? size : null,
                Pricing = Pricing(item)
            };
        }
    }

    protected override ModelPricing? Pricing(JsonElement model)
    {
        var pricing = Property(model, "pricing");
        var input = ProviderPrices.TokenPrice(Property(pricing, "input")) ?? 0;
        var output = ProviderPrices.TokenPrice(Property(pricing, "output")) ?? 0;
        if (input <= 0 && output <= 0) return null;
        var unit = String(pricing, "unit");
        if (unit?.Contains("M Token", StringComparison.OrdinalIgnoreCase) == true)
        { input /= 1_000_000m; output /= 1_000_000m; }
        else if (unit?.Contains("/ hour", StringComparison.OrdinalIgnoreCase) == true) output /= 3600m;
        return new() { Input = input, Output = output };
    }

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var form = new MultipartFormDataContent();
        var fields = Options(options, "model", "file", "stream");
        if (language is not null) fields["language"] = language;
        fields["model"] = model;
        AddFields(form, fields);
        AddFile(form, "file", audio, mediaType, fileName);
        using var client = CreateClient();
        using var upstream = await SendAsync(client, HttpMethod.Post, TranscriptionsPath, form, "Bearer", ApiKey(), headers, ct);
        var raw = await NativeHttp.ReadAsync<JsonElement>(upstream, Json, ct);
        return new() { Text = String(raw, "text") ?? throw new JsonException("BergetAI transcription response has no text."),
            Language = String(raw, "language") ?? language, Segments = [], ProviderMetadata = Metadata(raw),
            Response = new() { ModelId = model, Body = raw } };
    }

    public async Task<HttpResponseMessage> TranscribeNativeAsync(string model, IFormCollection request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Files.Count != 1 || request.Files.GetFile("file") is not { Length: > 0 } audio)
            throw new ProviderRequestException("BergetAI transcription requires one audio file.");
        var stream = string.Equals(request["stream"], "true", StringComparison.OrdinalIgnoreCase);
        using var form = new MultipartFormDataContent();
        foreach (var (name, values) in request)
        {
            if (Reserved(name, "model", "file", "stream") || stream && name.Equals("response_format", StringComparison.OrdinalIgnoreCase)) continue;
            foreach (var value in values) form.Add(new StringContent(value ?? ""), name);
        }
        form.Add(new StringContent(model), "model");
        AddFile(form, "file", audio);
        using var client = CreateClient();
        var upstream = await SendAsync(client, HttpMethod.Post, TranscriptionsPath, form, "Bearer", ApiKey(), headers, ct);
        if (!stream) return upstream;
        using (upstream)
        {
            var raw = await NativeHttp.ReadAsync<JsonElement>(upstream, Json, ct);
            var text = String(raw, "text") ?? throw new JsonException("BergetAI transcription response has no text.");
            var delta = text.Length == 0 ? "" : Frame(new { type = "transcript.text.delta", delta = text });
            return new(HttpStatusCode.OK) { Content = new StringContent(delta + Frame(new { type = "transcript.text.done", text }),
                Encoding.UTF8, "text/event-stream") };
        }

        static string Frame(object value) => $"data: {JsonSerializer.Serialize(value, Json)}\n\n";
    }

    public async Task<RerankingResponse> RerankAsync(RerankingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var docs = request.Documents?.Values ?? default;
        if (docs.ValueKind != JsonValueKind.Array || docs.GetArrayLength() == 0 || Array(docs).Any(x => x.ValueKind != JsonValueKind.String))
            throw new ProviderRequestException("BergetAI reranking requires a nonempty array of strings.");
        if (request.TopN is <= 0) throw new ProviderRequestException("topN must be positive.");
        var options = request.ProviderOptions?.GetValueOrDefault(ProviderId) ?? default;
        if (options.ValueKind is not (JsonValueKind.Undefined or JsonValueKind.Null or JsonValueKind.Object))
            throw new ProviderRequestException("BergetAI options must be an object.");
        var body = Options(options.ValueKind == JsonValueKind.Object ? options.EnumerateObject().Select(x => KeyValuePair.Create(x.Name, x.Value)) : null,
            "model", "query", "documents", "top_n");
        body["model"] = request.Model;
        body["query"] = request.Query;
        body["documents"] = Copy(docs);
        if (request.TopN is { } n) body["top_n"] = n;
        using var client = CreateClient();
        var (raw, upstream) = await NativeHttp.PostJsonAsync(client, "rerank", body, Bearer(ApiKey()), headers, Json, ct);
        var data = Property(raw, "data");
        if (data.ValueKind != JsonValueKind.Array) throw new JsonException("BergetAI rerank response has no data array.");
        var ranking = Array(data).Select(item => new RerankingRanking
        {
            Index = item.GetProperty("index").GetInt32(), RelevanceScore = item.GetProperty("relevance_score").GetDouble()
        }).OrderByDescending(item => item.RelevanceScore).Take(request.TopN ?? int.MaxValue).ToArray();
        return new() { Ranking = ranking, ProviderMetadata = Metadata(raw), Response = new() { Body = raw,
            Headers = NativeHttp.SafeResponseMetadataHeaders(upstream).Where(x => x.Value is not null).ToDictionary(x => x.Key, x => x.Value!),
            Id = String(raw, "id"), ModelId = String(raw, "model") ?? request.Model, Timestamp = DateTime.UtcNow } };
    }

    private Dictionary<string, JsonElement> Metadata(JsonElement raw) => new() { [ProviderId] = raw };

    private static bool Reserved(string field, params string[] controls) => controls.Contains(field, StringComparer.OrdinalIgnoreCase)
        || field.Equals("authorization", StringComparison.OrdinalIgnoreCase) || field.Equals("api_key", StringComparison.OrdinalIgnoreCase)
        || field.Equals("apiKey", StringComparison.OrdinalIgnoreCase) || field.Equals("x-api-key", StringComparison.OrdinalIgnoreCase)
        || field.Equals("headers", StringComparison.OrdinalIgnoreCase);

    private static JsonObject Options(IEnumerable<KeyValuePair<string, JsonElement>>? source, params string[] reserved)
    {
        var result = new JsonObject();
        foreach (var (field, value) in source ?? [])
            if (!Reserved(field, reserved) && value.ValueKind != JsonValueKind.Undefined) result[field] = Copy(value);
        return result;
    }
}
