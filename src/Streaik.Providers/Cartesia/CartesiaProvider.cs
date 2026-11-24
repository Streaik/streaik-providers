using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class CartesiaProviderRegistration
{
    public static IServiceCollection AddCartesia(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, CartesiaProvider.Id, "https://api.cartesia.ai/");
        return services.AddProvider<CartesiaProvider>(CartesiaProvider.Id, "Cartesia-Version");
    }
}

internal sealed partial class CartesiaProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, ISpeechProvider, IStreamingSpeechEventsProvider, IChatAudioRoutingProvider,
      ITranscriptionProvider, INativeAudioTranscriptionProvider, IRealtimeTokenProvider
{
    internal const string Id = "cartesia";
    private const string Version = "2026-08-14";
    private const string VersionHeader = "Cartesia-Version";
    private static readonly string[] SpeechModels = ["sonic-3.6", "sonic-3.5", "sonic-3", "sonic-latest"];
    private static readonly string[] BatchModels = ["ink-whisper", "ink-whisper-2025-06-04"];

    public bool IsSpeechModel(string model) => model.Split('/') is ["tts", var sonic, { Length: > 0 }] && SpeechModels.Contains(sonic);
    public bool IsTranscriptionModel(string model) => BatchModels.Contains(BatchModel(model), StringComparer.OrdinalIgnoreCase);
    private static string BatchModel(string model) => model.StartsWith("transcription/", StringComparison.Ordinal) ? model[14..] : model;

    private Action<System.Net.Http.Headers.HttpRequestHeaders> Auth(IReadOnlyDictionary<string, string> headers, string? version = null)
    {
        if (string.IsNullOrWhiteSpace(version)) version = headers.GetValueOrDefault(VersionHeader) ?? Version;
        if (!DateOnly.TryParseExact(version, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out _))
            throw new ProviderRequestException("Invalid Cartesia API version.");
        var key = RequiredKey(credentials, Id);
        return outgoing =>
        {
            outgoing.Authorization = new("Bearer", key);
            outgoing.TryAddWithoutValidation(VersionHeader, version);
        };
    }

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        foreach (var model in BatchModels) yield return Model("transcription/" + model, model, "transcription");
        foreach (var mode in new[] { "auto", "manual" })
            foreach (var model in new[] { "ink-2", "ink-preview", "ink-whisper" })
                if (mode != "auto" || model != "ink-whisper")
                    yield return Model($"transcription/realtime/{mode}/{model}", $"{model} realtime ({mode})", "transcription", ["realtime"]);

        using var client = clients.CreateClient(Id);
        await foreach (var voice in CatalogAsync(client, "voices", ct))
            if (String(voice, "id") is { Length: > 0 } id)
                foreach (var model in SpeechModels)
                    yield return Model($"tts/{model}/{id}", $"{model} {String(voice, "name") ?? id}", "speech",
                        new[] { "voice", "realtime-token", String(voice, "language"), String(voice, "gender") }.OfType<string>().ToArray());
        await foreach (var agent in CatalogAsync(client, "v1/agents", ct))
            if (String(agent, "id") is { Length: > 0 } id)
            {
                var model = Model($"agents/{id}", String(agent, "name") ?? id, "audio", ["realtime", "agent"]);
                model.Description = String(agent, "description");
                yield return model;
            }
    }

    private async IAsyncEnumerable<JsonElement> CatalogAsync(HttpClient client, string path, [EnumeratorCancellation] CancellationToken ct)
    {
        var auth = Auth(new Dictionary<string, string>());
        var cursors = new HashSet<string>(StringComparer.Ordinal);
        var ids = new HashSet<string>(StringComparer.Ordinal);
        string? cursor = null;
        do
        {
            var endpoint = path + "?limit=100" + (cursor is null ? "" : "&starting_after=" + Uri.EscapeDataString(cursor));
            JsonElement body;
            try { body = await NativeHttp.GetAsync<JsonElement>(client, endpoint, auth, ct); }
            catch (HttpRequestException ex) when (ex.StatusCode is not null) { yield break; } // Catalog permissions may differ from the media endpoints.
            string? last = null;
            foreach (var entry in Array(body.ValueKind == JsonValueKind.Array ? body : Property(body, "data")))
                if (String(entry, "id") is { Length: > 0 } id)
                {
                    last = id;
                    if (ids.Add(id)) yield return entry;
                }
            cursor = Property(body, "has_more").ValueKind == JsonValueKind.True ? String(body, "next_page") ?? last : null;
        } while (!string.IsNullOrWhiteSpace(cursor) && cursors.Add(cursor));
    }

    private static Model Model(string id, string name, string type, string[]? tags = null) => new() { Id = id, Name = name, OwnedBy = Id, Type = type, Tags = tags };

    public async Task<RealtimeTokenResponse> CreateRealtimeTokenAsync(string model, RealtimeTokenRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        RealtimeOptions.ValidateNativeConfig(request.SessionConfig);
        var parts = model.Split('/');
        var agent = parts is ["agents", { Length: > 0 }];
        var tts = parts is ["tts", var sonic, { Length: > 0 }] && SpeechModels.Contains(sonic);
        var stt = parts is ["transcription", "realtime", "auto" or "manual", "ink-2" or "ink-preview" or "ink-whisper"]
            && (parts[2] != "auto" || parts[3] != "ink-whisper");
        if (!(agent || tts || stt)) throw new ProviderRequestException("Use a Cartesia agent, voice, or realtime transcription model.");
        var options = RealtimeOptions.Read(request, Id);
        var seconds = 600;
        if (options.TryGetPropertyValue("expires_in", out var expiry) && (expiry is not JsonValue value || !value.TryGetValue(out seconds) || seconds is < 1 or > 3600))
            throw new ProviderRequestException("Cartesia expires_in must be between 1 and 3600 seconds.");
        if (options.Any(field => field.Key != "expires_in")) throw new ProviderRequestException("Cartesia token options support expires_in only.");
        var payload = new { grants = new { stt, agent, tts }, expires_in = seconds };
        using var client = clients.CreateClient(Id);
        var result = await NativeHttp.PostAsync<object, JsonElement>(client, "access-token", payload, Auth(headers, Version), headers, Json, ct);
        return new() { Token = RealtimeOptions.Required(result, "token", RequiredKey(credentials, Id)),
            ExpiresAt = DateTimeOffset.UtcNow.AddSeconds(seconds).ToUnixTimeSeconds(),
            Url = stt ? $"wss://api.cartesia.ai/stt/{(parts[2] == "auto" ? "turns/" : "")}websocket?cartesia_version={Version}&model={parts[3]}" :
                tts ? $"wss://api.cartesia.ai/tts/websocket?cartesia_version={Version}" :
                $"wss://api.cartesia.ai/v1/agents/websocket/{Uri.EscapeDataString(parts[1])}?cartesia_version={Version}" };
    }

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!IsTranscriptionModel(model)) throw new ProviderRequestException("Use a Cartesia batch transcription model.");
        model = BatchModel(model);
        var fields = options ?? new Dictionary<string, JsonElement>();
        var version = Option(fields, "apiVersion");
        var query = new List<string>();
        if (Option(fields, "encoding") is { Length: > 0 } encoding) query.Add("encoding=" + Uri.EscapeDataString(encoding));
        var rate = Field(fields, "sampleRate", "sample_rate");
        if (rate.ValueKind is not (JsonValueKind.Null or JsonValueKind.Undefined))
        {
            if (!int.TryParse(rate.ToString(), NumberStyles.None, CultureInfo.InvariantCulture, out var sampleRate) || sampleRate <= 0)
                throw new ProviderRequestException("Cartesia sample rate must be a positive integer.");
            query.Add("sample_rate=" + sampleRate.ToString(CultureInfo.InvariantCulture));
        }
        using var form = new MultipartFormDataContent();
        AddFile(form, "file", audio, mediaType, fileName);
        form.Add(new StringContent(model), "model");
        if (!string.IsNullOrWhiteSpace(language ?? Option(fields, "language"))) form.Add(new StringContent((language ?? Option(fields, "language"))!.Trim()), "language");
        var timestamps = Field(fields, "timestampGranularities", "timestamp_granularities[]", "timestamp_granularities");
        var granularities = timestamps.ValueKind == JsonValueKind.Array ? Array(timestamps) : [timestamps];
        foreach (var item in granularities)
            if (item.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(item.GetString()))
                form.Add(new StringContent(item.GetString()!), "timestamp_granularities[]");
        foreach (var (name, value) in fields)
            if (name is not ("model" or "file" or "language" or "apiVersion" or "encoding" or "sampleRate" or "sample_rate" or "timestampGranularities" or "timestamp_granularities" or "timestamp_granularities[]" or "stream" or "response_format"))
                form.Add(new StringContent(value.ValueKind == JsonValueKind.String ? value.GetString()! : value.GetRawText()), name);

        using var client = clients.CreateClient(Id);
        var path = query.Count == 0 ? "stt" : "stt?" + string.Join('&', query);
        using var outgoing = new HttpRequestMessage(HttpMethod.Post, path) { Content = form };
        Auth(headers, version)(outgoing.Headers);
        NativeHttp.ApplyHeaders(outgoing, headers);
        using var response = await NativeHttp.SendAsync(client, outgoing, ct);
        var body = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        if (String(body, "text") is not { } text) throw new JsonException("Cartesia returned no transcription text.");
        var words = Array(Property(body, "words")).Select(word => new TranscriptionSegment { Text = String(word, "word") ?? "",
            StartSecond = Number(word, "start") ?? 0, EndSecond = Number(word, "end") ?? Number(word, "start") ?? 0 })
            .Where(word => !string.IsNullOrWhiteSpace(word.Text)).ToArray();
        return new() { Text = text, Language = String(body, "language"), Segments = words, DurationInSeconds = Number(body, "duration"),
            ProviderMetadata = Metadata(body), Response = new() { Timestamp = DateTime.UtcNow, Headers = SafeHeaders(response), Body = body } };
    }

    public async Task<HttpResponseMessage> TranscribeNativeAsync(string model, IFormCollection request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var format = request["response_format"].ToString();
        if (format is not ("" or "json" or "text" or "verbose_json" or "srt" or "vtt")) throw new ProviderRequestException("Unsupported transcription response format.");
        var stream = request["stream"].ToString();
        if (stream is not ("" or "false" or "true")) throw new ProviderRequestException("Invalid transcription stream flag.");
        var file = request.Files.GetFile("file") ?? throw new ProviderRequestException("Expected an audio file.");
        var fields = request.Where(field => field.Key is not ("model" or "file" or "stream" or "response_format" or "language"))
            .ToDictionary(field => field.Key, field => field.Value.Count > 1 ? JsonSerializer.SerializeToElement(field.Value.ToArray()) : JsonSerializer.SerializeToElement(field.Value.ToString()), StringComparer.Ordinal);
        await using var input = file.OpenReadStream();
        var result = await TranscribeAsync(model, input, file.ContentType, file.FileName, request["language"], fields, headers, ct);
        if (stream == "true")
        {
            var delta = JsonSerializer.Serialize(new { type = "transcript.text.delta", delta = result.Text }, Json);
            var done = JsonSerializer.Serialize(new { type = "transcript.text.done", text = result.Text }, Json);
            return Reply((result.Text.Length == 0 ? "" : $"data: {delta}\n\n") + $"data: {done}\n\n", "text/event-stream");
        }
        return format switch
        {
            "text" => Reply(result.Text, "text/plain"),
            "verbose_json" => Reply(JsonSerializer.Serialize(new { text = result.Text, language = result.Language, duration = result.DurationInSeconds,
                segments = result.Segments.Select((item, index) => new { id = index, start = item.StartSecond, end = item.EndSecond, text = item.Text }) }, Json), "application/json"),
            "srt" or "vtt" => Reply(Captions(result, format == "vtt"), format == "vtt" ? "text/vtt" : "text/plain"),
            _ => Reply(JsonSerializer.Serialize(new { text = result.Text }, Json), "application/json")
        };
    }

    private static string Captions(TranscriptionResponse result, bool vtt)
    {
        var lines = result.Segments.ToArray();
        if (lines.Length == 0 && result.Text.Length > 0) lines = [new() { Text = result.Text, EndSecond = result.DurationInSeconds ?? 0 }];
        string Time(double seconds) => TimeSpan.FromSeconds(Math.Clamp(seconds, 0, 359999)).ToString(vtt ? @"hh\:mm\:ss\.fff" : @"hh\:mm\:ss\,fff", CultureInfo.InvariantCulture);
        return (vtt ? "WEBVTT\n\n" : "") + string.Join('\n', lines.Select((line, i) =>
            $"{i + 1}\n{Time(line.StartSecond)} --> {Time(line.EndSecond)}\n{line.Text}\n"));
    }

    private static HttpResponseMessage Reply(string body, string mediaType) => new(System.Net.HttpStatusCode.OK) { Content = new StringContent(body, Encoding.UTF8, mediaType) };
    private static JsonElement Field(IReadOnlyDictionary<string, JsonElement> fields, params string[] names)
        => names.Where(fields.ContainsKey).Select(name => fields[name]).FirstOrDefault();
    private static string? Option(IReadOnlyDictionary<string, JsonElement> fields, string name) => Field(fields, name) is { ValueKind: JsonValueKind.String } value ? value.GetString() : null;
    private static double? Number(JsonElement body, string name) => Property(body, name) is { ValueKind: JsonValueKind.Number } value && value.TryGetDouble(out var number) && double.IsFinite(number) ? number : null;
    private static Dictionary<string, string> SafeHeaders(HttpResponseMessage response) => response.Headers.Concat(response.Content.Headers)
        .Where(header => !header.Key.Equals("Set-Cookie", StringComparison.OrdinalIgnoreCase))
        .GroupBy(header => header.Key, StringComparer.OrdinalIgnoreCase)
        .ToDictionary(group => group.Key, group => string.Join(", ", group.SelectMany(header => header.Value)), StringComparer.OrdinalIgnoreCase);
    private static Dictionary<string, JsonElement> Metadata(JsonElement body) => new() { [Id] = body };
}
