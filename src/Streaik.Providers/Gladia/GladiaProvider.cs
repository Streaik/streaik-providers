using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Gladia;

public static class GladiaRegistration
{
    public static IServiceCollection AddGladia(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, GladiaProvider.Id, "https://api.gladia.io/");
        return services.AddProvider<GladiaProvider>(GladiaProvider.Id);
    }
}

internal sealed class GladiaProvider(IHttpClientFactory clients, IProviderCredentials credentials, ILogger<GladiaProvider> logger)
    : IModelListingProvider, ITranscriptionProvider, IRealtimeTokenProvider
{
    internal const string Id = "gladia";
    private const string LiveModel = "solaria-1";
    private const string Prerecorded = "v2/pre-recorded";
    private static readonly string[] Models = [LiveModel, "solaria-3"];

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        foreach (var model in Models)
        {
            ct.ThrowIfCancellationRequested();
            var live = model == LiveModel;
            yield return new Model { Id = model, Name = live ? "Solaria 1" : "Solaria 3", Type = "transcription", OwnedBy = Id,
                Tags = live ? ["realtime"] : null };
        }
    }

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!Models.Contains(model, StringComparer.Ordinal)) throw new ProviderRequestException("Unknown Gladia transcription model.");
        var payload = new JsonObject();
        foreach (var (name, value) in options ?? new Dictionary<string, JsonElement>())
        {
            // This adapter always operates on the submitted file, without asynchronous callbacks.
            if (name is "model" or "file" or "audio_url" or "language" or "callback" or "callback_url" or "callback_config" or "webhook_config") continue;
            if (value.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined) continue;
            if (name == "prompt") throw new ProviderRequestException("Use Gladia custom_vocabulary instead of prompt.", 422);
            payload[name] = NormalizeOption(name, value);
        }
        if (!string.IsNullOrWhiteSpace(language))
        {
            if (payload["language_config"] is null) payload["language_config"] = new JsonObject();
            if (payload["language_config"] is not JsonObject config) throw new ProviderRequestException("language_config must be an object.", 422);
            config["languages"] = new JsonArray(JsonValue.Create(language));
        }
        payload["model"] = model;

        var key = RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        using var form = new MultipartFormDataContent();
        AddFile(form, "audio", audio, mediaType, fileName);
        var upload = await SendJsonAsync(client, HttpMethod.Post, "v2/upload", form, key, ct);
        payload["audio_url"] = String(upload, "audio_url") ?? throw new JsonException("Missing Gladia upload URL.");
        var job = await SendJsonAsync(client, HttpMethod.Post, Prerecorded, JsonContent.Create(payload, options: Json), key, ct);
        var id = String(job, "id");
        if (id is null || !Guid.TryParse(id, out _)) throw new JsonException("Invalid Gladia job ID.");
        var path = $"{Prerecorded}/{Uri.EscapeDataString(id)}";
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromMinutes(5));
        try
        {
            var delay = TimeSpan.FromSeconds(1);
            while (true)
            {
                var status = await SendJsonAsync(client, HttpMethod.Get, path, null, key, deadline.Token);
                switch (String(status, "status"))
                {
                    case "done": return MapTranscript(status, id);
                    case "error": throw new ProviderRequestException("Gladia could not transcribe the submitted audio.", 422);
                    case "queued": case "processing": break;
                    default: throw new JsonException("Unknown Gladia transcription status.");
                }
                await Task.Delay(delay, deadline.Token);
                delay = TimeSpan.FromMilliseconds(Math.Min(delay.TotalMilliseconds * 1.5, 5000));
            }
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested && deadline.IsCancellationRequested)
        { throw new ProviderRequestException("Gladia transcription did not complete within five minutes.", 504); }
        finally
        {
            try
            {
                using var cleanup = new CancellationTokenSource(TimeSpan.FromSeconds(5));
                using var response = await SendAsync(client, HttpMethod.Delete, path, null, key, cleanup.Token);
            }
            catch (Exception ex) when (ex is HttpRequestException or OperationCanceledException or IOException)
            { logger.LogWarning("Could not delete a completed or abandoned Gladia transcription job."); }
        }
    }

    public async Task<RealtimeTokenResponse> CreateRealtimeTokenAsync(string model, RealtimeTokenRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (model != LiveModel) throw new ProviderRequestException("Use gladia/solaria-1 for live transcription.");
        RealtimeOptions.ValidateNativeConfig(request.SessionConfig);
        var payload = RealtimeOptions.Read(request, Id);
        foreach (var name in new[] { "callback", "callback_url", "callback_config", "webhook_config", "model" }) payload.Remove(name);
        payload["encoding"] ??= "wav/pcm";
        payload["bit_depth"] ??= 16;
        payload["sample_rate"] ??= 16000;
        payload["channels"] ??= 1;
        using var client = clients.CreateClient(Id);
        var key = RequiredKey(credentials, Id);
        var response = await SendJsonAsync(client, HttpMethod.Post, "v2/live", JsonContent.Create(payload, options: Json), key, ct);
        var url = RealtimeOptions.Required(response, "url", key);
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) || uri.Scheme != "wss" || uri.UserInfo.Length > 0
            || !(uri.Host == "gladia.io" || uri.Host.EndsWith(".gladia.io", StringComparison.OrdinalIgnoreCase)))
            throw new JsonException("Invalid Gladia WebSocket URL.");
        return new() { Token = RealtimeOptions.Required(response, "id", key), Url = url };
    }

    private static TranscriptionResponse MapTranscript(JsonElement root, string id)
    {
        var result = Property(root, "result");
        var transcript = Property(result, "transcription");
        var metadata = Property(result, "metadata");
        var utterances = Array(Property(transcript, "utterances")).ToArray();
        var language = Array(Property(transcript, "languages")).FirstOrDefault(x => x.ValueKind == JsonValueKind.String);
        var details = new JsonObject { ["id"] = id };
        foreach (var property in result.ValueKind == JsonValueKind.Object ? result.EnumerateObject().ToArray() : [])
            if (property.Name != "transcription") details[property.Name] = Copy(property.Value);
        foreach (var name in new[] { "languages", "utterances", "subtitles" })
            if (Property(transcript, name) is { ValueKind: not JsonValueKind.Undefined } value) details[name] = Copy(value);
        return new TranscriptionResponse
        {
            Text = String(transcript, "full_transcript") ?? throw new JsonException("Missing Gladia transcript."),
            Language = language.ValueKind == JsonValueKind.String ? language.GetString() : utterances.Select(x => String(x, "language")).FirstOrDefault(x => x is not null),
            DurationInSeconds = Number(metadata, "audio_duration"),
            Segments = utterances.Where(x => String(x, "text") is not null).Select(x => new TranscriptionSegment
            { Text = String(x, "text")!, StartSecond = Number(x, "start") ?? 0, EndSecond = Number(x, "end") ?? Number(x, "start") ?? 0 }).ToArray(),
            ProviderMetadata = new() { [Id] = JsonSerializer.SerializeToElement(details, Json) },
            Response = new() { Timestamp = DateTime.UtcNow }
        };
    }

    private static double? Number(JsonElement element, string name) => Property(element, name) is { ValueKind: JsonValueKind.Number } value
        && value.TryGetDouble(out var number) && double.IsFinite(number) ? number : null;

    private static JsonNode? NormalizeOption(string name, JsonElement value)
    {
        if (value.ValueKind == JsonValueKind.String && value.GetString() is { } text &&
            (name.EndsWith("_config", StringComparison.Ordinal) || name is "diarization" or "subtitles" or "translation"
                or "summarization" or "sentences" or "custom_vocabulary" or "custom_metadata"))
        {
            try { return JsonNode.Parse(text); }
            catch (JsonException) { throw new ProviderRequestException($"Gladia option '{name}' requires a JSON value.", 422); }
        }
        return Copy(value);
    }

    private static async Task<JsonElement> SendJsonAsync(HttpClient client, HttpMethod method, string path, HttpContent? content, string key, CancellationToken ct)
    {
        using var response = await SendAsync(client, method, path, content, key, ct);
        return await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
    }

    private static async Task<HttpResponseMessage> SendAsync(HttpClient client, HttpMethod method, string path, HttpContent? content, string key, CancellationToken ct)
    {
        using var request = new HttpRequestMessage(method, path) { Content = content };
        request.Headers.Add("x-gladia-key", key);
        return await NativeHttp.SendAsync(client, request, ct);
    }
}
