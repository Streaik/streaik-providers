using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Streaik.Core.Models;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class YourVoicProviderRegistration
{
    public static IServiceCollection AddYourVoic(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, YourVoicProvider.Id, "https://yourvoic.com/api/v1/");
        return services.AddProvider<YourVoicProvider>(YourVoicProvider.Id);
    }
}

internal sealed class YourVoicProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IStreamingSpeechProvider, IChatAudioRoutingProvider, ITranscriptionProvider, INativeAudioTranscriptionProvider
{
    internal const string Id = "yourvoic";
    private static readonly HashSet<string> SpeechReserved = new(StringComparer.OrdinalIgnoreCase)
        { "model", "input", "text", "response_format", "stream_format", "stream", "instructions" };
    private static readonly HashSet<string> TranscriptReserved = new(StringComparer.OrdinalIgnoreCase)
        { "model", "file", "stream", "response_format" };
    private static readonly HashSet<string> CipherFields = new(StringComparer.OrdinalIgnoreCase)
        { "prompt", "timestamp_granularities", "timestamp_granularities[]" };
    private static readonly HashSet<string> LucidFields = new(StringComparer.OrdinalIgnoreCase)
        { "diarize", "smart_format", "punctuate", "keywords" };

    public IAsyncEnumerable<Model> ListModelsAsync(CancellationToken ct) => StaticModelListing.ListAsync(Id, credentials.GetApiKey(Id), ct);
    public bool IsSpeechModel(string model) => Family(model, "aura-") || Family(model, "rapid-");
    public bool IsTranscriptionModel(string model) => Family(model, "cipher-") || Family(model, "lucid-");

    // Upstream generates a single audio body; the gateway adds framing for its streaming clients.
    public Task<HttpResponseMessage> SpeakStreamingAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeakAsync(request, headers, ct);

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (IsTranscriptionModel(request.Model)) throw new ProviderRequestException("This YourVoic model does not generate speech.", 422);
        if (string.IsNullOrWhiteSpace(request.Input)) throw new ProviderRequestException("YourVoic requires nonempty text.");
        var options = NativeMediaOptions.Read(Id, request.AdditionalProperties, SpeechReserved);
        var fields = new Dictionary<string, object?>(StringComparer.Ordinal);
        var format = (request.ResponseFormat ?? StringOption(options, "format"))?.Trim().ToLowerInvariant() switch
        {
            "mpeg" => "mp3", "wave" => "wav", "" => null, var value => value
        };
        foreach (var (name, value) in options)
            if (!name.Equals("format", StringComparison.OrdinalIgnoreCase)
                && !(request.Voice is not null && name.Equals("voice", StringComparison.OrdinalIgnoreCase))
                && !(request.Speed.HasValue && name.Equals("speed", StringComparison.OrdinalIgnoreCase))) fields[name] = value;
        fields["text"] = request.Input;
        fields["model"] = request.Model;
        if (request.Voice is { Length: > 0 }) fields["voice"] = request.Voice;
        if (request.Speed is { } speed) fields["speed"] = speed;
        if (format is not null) fields["format"] = format;

        using var client = clients.CreateClient(Id);
        using var message = Request("tts/generate", JsonContent.Create(fields, options: Json), headers);
        message.Headers.Accept.ParseAdd("audio/*");
        using var upstream = await NativeHttp.SendAsync(client, message, ct);
        var audio = upstream.Content;
        var mime = audio.Headers.ContentType?.MediaType;
        if (mime is null or "application/octet-stream")
            audio.Headers.ContentType = new(format switch
            {
                "wav" => "audio/wav", "flac" => "audio/flac", "opus" or "ogg" => "audio/ogg", "aac" => "audio/aac",
                "pcm" or "pcm16" => "application/octet-stream", _ => "audio/mpeg"
            });
        else if (!mime.StartsWith("audio/", StringComparison.OrdinalIgnoreCase)) throw new IOException("YourVoic did not return audio.");
        var warnings = string.IsNullOrWhiteSpace(request.Instructions) ? System.Array.Empty<object>()
            : [new { type = "unsupported", feature = "instructions" }];
        var result = new SpeechProviderResponse(audio, NativeMediaOptions.Metadata(Id, new { headers = SafeHeaders(upstream), warnings }));
        upstream.Content = null; // Transfer audio to the gateway response without buffering it here.
        return result;
    }

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var native = NativeMediaOptions.Read(Id, options, TranscriptReserved);
        var requestedFormat = options?.FirstOrDefault(pair => pair.Key.Equals("response_format", StringComparison.OrdinalIgnoreCase)).Value;
        var format = Format(requestedFormat is { ValueKind: JsonValueKind.String } value ? value.GetString() : null);
        using var response = await TranscribeUpstreamAsync(model, audio, mediaType, fileName, language, format, native, headers, ct);
        var (raw, text) = await ReadTranscriptAsync(response, format, model, ct);
        var upstreamHeaders = SafeHeaders(response);
        var segments = new List<TranscriptionSegment>();
        foreach (var item in Array(TranscriptItems(raw)))
        {
            if ((String(item, "text") ?? String(item, "word")) is not { Length: > 0 } segmentText) continue;
            if (Property(item, "speaker") is { ValueKind: JsonValueKind.String or JsonValueKind.Number } speaker)
                segmentText = $"speaker_{speaker}: {segmentText}";
            var start = Number(item, "start") ?? 0;
            segments.Add(new() { Text = segmentText, StartSecond = start, EndSecond = Number(item, "end") ?? start });
        }
        var ignored = Family(model, "cipher-") ? LucidFields : Family(model, "lucid-") ? CipherFields : null;
        var warnings = native.Where(pair => ignored?.Contains(pair.Key) == true && pair.Value.ValueKind != JsonValueKind.Null)
            .Select(pair => new { type = "ignored", feature = pair.Key, reason = "Not supported by this YourVoic model family." }).ToArray();
        return new()
        {
            Text = text, Language = String(raw, "language") ?? language, DurationInSeconds = Number(raw, "duration"), Segments = segments,
            Warnings = warnings,
            ProviderMetadata = NativeMediaOptions.Metadata(Id, new { body = raw, headers = upstreamHeaders }),
            Response = new() { ModelId = model, Body = raw, Headers = upstreamHeaders }
        };
    }

    public async Task<HttpResponseMessage> TranscribeNativeAsync(string model, IFormCollection request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var file = request.Files.GetFile("file") ?? throw new ProviderRequestException("YourVoic requires an audio file.");
        var format = Format(request["response_format"]);
        var stream = string.Equals(request["stream"], "true", StringComparison.OrdinalIgnoreCase);
        if (stream && format != "json") throw new ProviderRequestException("Streaming transcription requires JSON response format.");
        var fields = request.ToDictionary(pair => pair.Key, pair => pair.Key is "providerOptions" or Id
            ? JsonSerializer.Deserialize<JsonElement>(pair.Value.ToString()) : JsonSerializer.SerializeToElement(pair.Value.ToString(), Json));
        var options = NativeMediaOptions.Read(Id, fields, TranscriptReserved);
        await using var audio = file.OpenReadStream();
        using var upstream = await TranscribeUpstreamAsync(model, audio, file.ContentType, file.FileName, request["language"], format, options, headers, ct);
        var (raw, text) = await ReadTranscriptAsync(upstream, format, model, ct);
        HttpContent content = format switch
        {
            "text" or "srt" => new StringContent(text, Encoding.UTF8, "text/plain"),
            "vtt" => new StringContent(text, Encoding.UTF8, "text/vtt"),
            "verbose_json" => JsonContent.Create(Verbose(raw, text), options: Json),
            _ => JsonContent.Create(new OpenAITranscriptionResponse { Text = text }, options: Json)
        };
        if (stream)
        {
            content.Dispose();
            var delta = text.Length == 0 ? "" : $"data: {JsonSerializer.Serialize(new OpenAITranscriptionTextDelta { Delta = text }, Json)}\n\n";
            var done = JsonSerializer.Serialize(new OpenAITranscriptionTextDone { Text = text }, Json);
            content = new StringContent($"{delta}data: {done}\n\n", Encoding.UTF8, "text/event-stream");
        }
        return new(System.Net.HttpStatusCode.OK) { Content = content };
    }

    private async Task<HttpResponseMessage> TranscribeUpstreamAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, string format, Dictionary<string, JsonElement> options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (IsSpeechModel(model)) throw new ProviderRequestException("This YourVoic model does not transcribe audio.", 422);
        var cipher = Family(model, "cipher-");
        var lucid = Family(model, "lucid-");
        if (lucid && format is "srt" or "vtt") throw new ProviderRequestException("Lucid does not return subtitle formats.", 422);
        using var form = new MultipartFormDataContent();
        AddFile(form, "file", audio, mediaType, fileName);
        Add("model", model);
        Add("language", string.IsNullOrWhiteSpace(language) ? StringOption(options, "language") : language);
        if (!lucid) Add("response_format", format);
        var timestamps = new List<string>();
        foreach (var (name, value) in options)
        {
            if (name.Equals("language", StringComparison.OrdinalIgnoreCase) || value.ValueKind == JsonValueKind.Null
                || lucid && CipherFields.Contains(name) || cipher && LucidFields.Contains(name)) continue;
            if (name.Equals("timestamp_granularities", StringComparison.OrdinalIgnoreCase)
                || name.Equals("timestamp_granularities[]", StringComparison.OrdinalIgnoreCase))
            {
                if (value.ValueKind == JsonValueKind.Array) timestamps.AddRange(Array(value).Select(FormValue));
                else timestamps.Add(FormValue(value));
            }
            else Add(name, FormValue(value));
        }
        if (timestamps.Count > 0) Add("timestamp_granularities", string.Join(',', timestamps));
        var path = cipher ? "stt/cipher/transcribe" : lucid ? "stt/lucid/transcribe" : "stt/transcribe";
        using var client = clients.CreateClient(Id);
        using var message = Request(path, form, headers);
        return await NativeHttp.SendAsync(client, message, ct);

        void Add(string name, string? value)
        {
            if (!string.IsNullOrWhiteSpace(value)) form.Add(new StringContent(value, Encoding.UTF8), name);
        }
    }

    private HttpRequestMessage Request(string path, HttpContent content, IReadOnlyDictionary<string, string> headers)
    {
        var key = RequiredKey(credentials, Id);
        var message = new HttpRequestMessage(HttpMethod.Post, path) { Content = content };
        message.Headers.Add("X-API-Key", key);
        NativeHttp.ApplyHeaders(message, headers);
        return message;
    }

    private static async Task<(JsonElement Raw, string Text)> ReadTranscriptAsync(HttpResponseMessage response, string format, string model, CancellationToken ct)
    {
        if (format is "srt" or "vtt" || format == "text" && !Family(model, "lucid-")
            && response.Content.Headers.ContentType?.MediaType != "application/json")
        {
            var text = await response.Content.ReadAsStringAsync(ct);
            return (JsonSerializer.SerializeToElement(text, Json), text);
        }
        var raw = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        return (raw, String(raw, "text") ?? throw new JsonException("YourVoic did not return transcript text."));
    }

    private static OpenAITranscriptionVerboseResponse Verbose(JsonElement body, string text)
    {
        var duration = Number(body, "duration") ?? 0;
        return new() { Text = text, Language = String(body, "language") ?? "", Duration = duration,
            Usage = new OpenAITranscriptionDurationUsage { Seconds = duration },
            Segments = Property(body, "segments") is { ValueKind: JsonValueKind.Array } segments
                ? Array(segments).Select((part, index) => new OpenAITranscriptionSegment
                {
                    Id = (int)(Number(part, "id") ?? index), Seek = (int)(Number(part, "seek") ?? 0),
                    Start = Number(part, "start") ?? 0, End = Number(part, "end") ?? 0, Text = String(part, "text") ?? "",
                    Tokens = Array(Property(part, "tokens")).Where(item => item.ValueKind == JsonValueKind.Number && item.TryGetInt32(out _)).Select(item => item.GetInt32()).ToArray(),
                    Temperature = (float)(Number(part, "temperature") ?? 0), AverageLogprob = (float)(Number(part, "avg_logprob") ?? 0),
                    CompressionRatio = (float)(Number(part, "compression_ratio") ?? 0), NoSpeechProbability = (float)(Number(part, "no_speech_prob") ?? 0)
                }).ToArray() : null,
            Words = Property(body, "words") is { ValueKind: JsonValueKind.Array } words
                ? Array(words).Select(word => new OpenAITranscriptionWord
                { Word = String(word, "word") ?? "", Start = Number(word, "start") ?? 0, End = Number(word, "end") ?? 0 }).ToArray() : null };
    }

    private static JsonElement TranscriptItems(JsonElement raw) => new[] { "utterances", "segments", "words" }
        .Select(name => Property(raw, name)).FirstOrDefault(item => item.ValueKind == JsonValueKind.Array);

    private static string Format(string? value) => value?.Trim().ToLowerInvariant() switch
    {
        null or "" => "json", "json" or "text" or "verbose_json" or "srt" or "vtt" => value.Trim().ToLowerInvariant(),
        _ => throw new ProviderRequestException("Unsupported transcription response format.")
    };

    private static bool Family(string model, string prefix) => model.StartsWith(prefix, StringComparison.OrdinalIgnoreCase);
    private static string FormValue(JsonElement value) => value.ValueKind == JsonValueKind.String ? value.GetString()! : value.GetRawText();
    private static string? StringOption(Dictionary<string, JsonElement> options, string name)
        => options.FirstOrDefault(pair => pair.Key.Equals(name, StringComparison.OrdinalIgnoreCase)).Value is { ValueKind: JsonValueKind.String } value
            ? value.GetString() : null;
    private static double? Number(JsonElement value, string name)
        => Property(value, name) is { ValueKind: JsonValueKind.Number } number && number.TryGetDouble(out var parsed) && double.IsFinite(parsed) ? parsed : null;

    private static Dictionary<string, string> SafeHeaders(HttpResponseMessage response)
        => NativeHttp.SafeResponseMetadataHeaders(response.Headers.Concat(response.Content.Headers)
            .GroupBy(header => header.Key, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => (string?)string.Join(", ", group.SelectMany(header => header.Value)), StringComparer.OrdinalIgnoreCase))
            .Where(header => header.Value is not null).ToDictionary(header => header.Key, header => header.Value!, StringComparer.OrdinalIgnoreCase);
}
