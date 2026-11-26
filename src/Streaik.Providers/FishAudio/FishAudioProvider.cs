using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class FishAudioRegistration
{
    public static IServiceCollection AddFishAudio(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, "fishaudio", "https://api.fish.audio/");
        return services.AddProvider<FishAudioProvider>("fishaudio");
    }
}

internal sealed partial class FishAudioProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, ISpeechProvider, IStreamingSpeechEventsProvider, ITranscriptionProvider,
      IChatAudioRoutingProvider, IRealtimeTokenProvider, IOpenAILiveSessionProvider
{
    private const string Id = "fishaudio";
    private static readonly HashSet<string> SpeechOptions = new(StringComparer.Ordinal)
    {
        "reference_id", "references", "prosody", "language", "format", "sample_rate", "mp3_bitrate", "opus_bitrate",
        "temperature", "top_p", "chunk_length", "normalize", "latency", "max_new_tokens", "repetition_penalty",
        "min_chunk_length", "condition_on_previous_chunks", "early_stop_threshold", "features", "pronunciation_dictionary"
    };

    public bool IsSpeechModel(string model) => model is "s1" or "s2-pro" or "s2.1-pro" or "s2.1-pro-free" or "drama-3";
    public bool IsTranscriptionModel(string model) => model is "asr" or "transcribe-1-pro";

    private HttpRequestMessage Request(HttpMethod method, string path, IReadOnlyDictionary<string, string> headers,
        string? model = null, string accept = "application/json")
    {
        var key = RequiredKey(credentials, Id);
        var request = new HttpRequestMessage(method, path);
        request.Headers.Authorization = new("Bearer", key);
        request.Headers.Accept.ParseAdd(accept);
        if (model is not null) request.Headers.Add("model", model);
        NativeHttp.ApplyHeaders(request, headers);
        return request;
    }

    private static Dictionary<string, JsonElement> Options(IReadOnlyDictionary<string, JsonElement>? extras, bool flat = false)
    {
        var options = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        if (extras?.TryGetValue(Id, out var native) == true && native.ValueKind != JsonValueKind.Null)
        {
            if (native.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Fish Audio options must be an object.");
            foreach (var field in native.EnumerateObject()) options[field.Name] = field.Value;
        }
        if (flat && extras is not null)
            foreach (var (name, value) in extras)
                if (name != Id) options[name] = value;
        return options;
    }

    private static Dictionary<string, JsonElement> Metadata(JsonElement value) => new() { [Id] = value };

    private static Dictionary<string, string> ResponseHeaders(HttpResponseMessage response)
        => response.Headers.Concat(response.Content.Headers)
            .Where(h => h.Key.ToLowerInvariant() is "content-type" or "request-id" or "x-request-id" or "retry-after"
                || h.Key.StartsWith("x-ratelimit-", StringComparison.OrdinalIgnoreCase))
            .GroupBy(h => h.Key, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(g => g.Key.ToLowerInvariant(), g => string.Join(", ", g.SelectMany(h => h.Value)));

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var plan = SpeechPlan(request.Model, request.Input, request.Voice, request.ResponseFormat, request.Speed, null, null,
            Options(request.AdditionalProperties, flat: true), false);
        using var client = clients.CreateClient(Id);
        using var message = Request(HttpMethod.Post, "v1/tts", headers, request.Model, plan.Mime);
        message.Content = JsonContent.Create(plan.Body, options: Json);
        using var response = await NativeHttp.SendAsync(client, message, ct);
        if (response.Content.Headers.ContentType?.MediaType is null or "application/octet-stream" || plan.Mime == "audio/pcm")
            response.Content.Headers.ContentType = MediaTypeHeaderValue.Parse(plan.Mime);
        var result = new SpeechProviderResponse(response.Content,
            Metadata(JsonSerializer.SerializeToElement(new { headers = ResponseHeaders(response) }, Json)));
        response.Content = null; // The caller now owns the upstream audio content.
        return result;
    }

    private sealed record SpeechPlanData(Dictionary<string, object?> Body, string Mime, int Rate);

    private SpeechPlanData SpeechPlan(string model, string text, string? voice, string? outputFormat, double? speed,
        string? language, double? rate, Dictionary<string, JsonElement> options, bool streaming)
    {
        if (!IsSpeechModel(model)) throw new ProviderRequestException("A Fish Audio speech model is required.", 422);
        if (string.IsNullOrWhiteSpace(text)) throw new ProviderRequestException("Speech text is required.");
        if (options.TryGetValue("references", out var references) && references.ValueKind != JsonValueKind.Null)
            throw new ProviderRequestException("Use a Fish Audio reference_id or voice; inline references require MessagePack.", 422);
        var body = options.Where(p => SpeechOptions.Contains(p.Key)).ToDictionary(p => p.Key, p => (object?)p.Value);
        body["text"] = text;
        if (voice is not null) body["reference_id"] = voice;
        if (language is not null) body["language"] = language;
        var formatValue = options.GetValueOrDefault("format");
        if (outputFormat is null && formatValue.ValueKind is not (JsonValueKind.String or JsonValueKind.Null or JsonValueKind.Undefined))
            throw new ProviderRequestException("Fish Audio format must be a string.");
        var format = (outputFormat ?? (formatValue.ValueKind == JsonValueKind.String ? formatValue.GetString() : null))?.ToLowerInvariant() switch
        {
            null or "mp3" or "audio/mpeg" or "audio/mp3" => ("mp3", "audio/mpeg"),
            "wav" or "audio/wav" or "audio/x-wav" => ("wav", "audio/wav"),
            "pcm" or "pcm16" or "raw" or "audio/pcm" or "audio/l16" => ("pcm", "audio/pcm"),
            "opus" or "ogg" or "audio/opus" or "audio/ogg" => ("opus", "audio/ogg"),
            _ => throw new ProviderRequestException("Unsupported Fish Audio speech format.", 422)
        };
        body["format"] = format.Item1;
        if (streaming) body.TryAdd("latency", "balanced");
        if (speed is { } value)
        {
            if (!double.IsFinite(value) || value is < 0.5 or > 2) throw new ProviderRequestException("Fish Audio speed must be between 0.5 and 2.0.");
            var raw = options.GetValueOrDefault("prosody");
            var prosody = raw.ValueKind == JsonValueKind.Object
                ? raw.EnumerateObject().ToDictionary(p => p.Name, p => (object?)p.Value) : new();
            prosody["speed"] = value;
            body["prosody"] = prosody;
        }
        if (rate is null && options.TryGetValue("sample_rate", out var requested) && requested.ValueKind != JsonValueKind.Null)
        {
            if (requested.ValueKind != JsonValueKind.Number || !requested.TryGetInt32(out var valueRate))
                throw new ProviderRequestException("Fish Audio sample_rate must be an integer.");
            rate = valueRate;
        }
        if (rate is not null)
        {
            int[] rates = format.Item1 switch { "opus" => [48000], "mp3" => [32000, 44100], _ => [8000, 16000, 24000, 32000, 44100] };
            if (!rates.Any(r => r == rate)) throw new ProviderRequestException("Fish Audio sample rate is not supported for the selected format.");
            body["sample_rate"] = (int)rate.Value;
        }
        return new(body, format.Item2, (int?)rate ?? (format.Item1 == "opus" ? 48000 : 44100));
    }

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeechEventsAsync(request.Model, request.Instructions, SpeechPlan(request.Model, request.Input, request.Voice, request.ResponseFormat,
            request.Speed, null, null, Options(request.AdditionalProperties, flat: true), true), false, headers, ct);

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Text is not { Length: > 0 } || request.Text.Any(p => p?.Text is null || p.Type != "text"))
            throw new ProviderRequestException("At least one valid speech text part is required.");
        var text = string.Concat(request.Text.Select(p => p.Text));
        SpeechPlanData? plan = null;
        foreach (var part in request.Text)
        {
            var options = Options(request.ProviderOptions);
            foreach (var field in Options(part.ProviderOptions)) options[field.Key] = field.Value;
            var current = SpeechPlan(request.Model, text, part.Voice ?? request.Voice, request.OutputFormat?.Type,
                request.Speed, request.Language, request.OutputFormat?.Rate, options, true);
            if (plan is not null && !JsonElement.DeepEquals(JsonSerializer.SerializeToElement(plan.Body, Json), JsonSerializer.SerializeToElement(current.Body, Json)))
                throw new ProviderRequestException("Fish Audio requires one effective voice and options configuration across text parts.", 422);
            plan ??= current;
        }
        return SpeechEventsAsync(request.Model, request.Instructions, plan!, request.IncludeRawChunks == true, headers, ct);
    }

    private async IAsyncEnumerable<StreamingSpeechPart> SpeechEventsAsync(string model, string? instructions, SpeechPlanData plan, bool raw,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        using var message = Request(HttpMethod.Post, "v1/tts/stream/with-timestamp", headers, model, "text/event-stream");
        message.Content = JsonContent.Create(plan.Body, options: Json);
        using var response = await NativeHttp.SendAsync(client, message, ct);
        if (response.Content.Headers.ContentType?.MediaType != "text/event-stream") throw new IOException("Fish Audio did not return an event stream.");
        yield return new SpeechStreamStartPart { Warnings = string.IsNullOrWhiteSpace(instructions) ? [] : [new { type = "unsupported", feature = "instructions" }] };
        yield return new SpeechResponseMetadataPart { Timestamp = DateTime.UtcNow, ModelId = model, Headers = ResponseHeaders(response) };
        var alignments = new SortedDictionary<int, JsonElement>();
        await foreach (var item in NativeHttp.ReadSseAsync<JsonElement>(response, Json, _ => false, ct))
        {
            if (item.ValueKind != JsonValueKind.Object || Property(item, "error").ValueKind != JsonValueKind.Undefined)
                throw new IOException("Fish Audio returned an invalid speech event or an upstream error.");
            var encoded = String(item, "audio_base64") ?? throw new IOException("Fish Audio speech event is missing audio_base64.");
            byte[] audio;
            try { audio = Convert.FromBase64String(encoded); }
            catch (FormatException) { throw new IOException("Fish Audio returned invalid base64 audio."); }
            if (raw) yield return new SpeechRawPart { RawValue = item };
            var fields = item.EnumerateObject().Where(p => p.Name != "audio_base64").ToDictionary(p => p.Name, p => p.Value);
            var alignment = Property(item, "alignment");
            var seq = Property(item, "chunk_seq");
            if (alignment.ValueKind == JsonValueKind.Object && seq.ValueKind == JsonValueKind.Number && seq.TryGetInt32(out var index))
                alignments[index] = JsonSerializer.SerializeToElement(new { chunk_seq = index, alignment }, Json);
            yield return new SpeechAudioPart { Audio = Convert.ToBase64String(audio), AudioFormat = new() { Type = plan.Mime, Rate = plan.Rate },
                ProviderMetadata = Metadata(JsonSerializer.SerializeToElement(fields, Json)) };
        }
        yield return new SpeechFinishPart { ProviderMetadata = Metadata(JsonSerializer.SerializeToElement(new { alignments = alignments.Values }, Json)) };
    }

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!IsTranscriptionModel(model)) throw new ProviderRequestException("A Fish Audio transcription model is required.", 422);
        var fields = Options(options, flat: true);
        using var client = clients.CreateClient(Id);
        using var message = Request(HttpMethod.Post, "v1/asr", headers, "transcribe-1-pro");
        using var form = new MultipartFormDataContent();
        AddFile(form, "audio", audio, mediaType, fileName);
        var nativeLanguage = fields.GetValueOrDefault("language");
        language ??= nativeLanguage.ValueKind == JsonValueKind.String ? nativeLanguage.GetString() : null;
        if (!string.IsNullOrWhiteSpace(language)) form.Add(new StringContent(language), "language");
        if (fields.TryGetValue("ignore_timestamps", out var timestamps))
        {
            var value = timestamps.ValueKind == JsonValueKind.String ? timestamps.GetString() : timestamps.GetRawText();
            if (!bool.TryParse(value, out var ignore)) throw new ProviderRequestException("Fish Audio ignore_timestamps must be a boolean.");
            form.Add(new StringContent(ignore ? "true" : "false"), "ignore_timestamps");
        }
        message.Content = form;
        using var response = await NativeHttp.SendAsync(client, message, ct);
        var root = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        var text = String(root, "text") ?? throw new JsonException("Fish Audio transcription response is missing text.");
        return new() { Text = text, Language = String(root, "language") ?? language, DurationInSeconds = Number(root, "duration"),
            Segments = Array(Property(root, "segments")).Select(item => new TranscriptionSegment { Text = String(item, "text") ?? "",
                StartSecond = Number(item, "start") ?? 0, EndSecond = Number(item, "end") ?? 0 }).Where(p => p.Text.Length > 0).ToArray(),
            ProviderMetadata = Metadata(root), Response = new() { Timestamp = DateTime.UtcNow, ModelId = model, Headers = ResponseHeaders(response) } };
    }

    private static double? Number(JsonElement value, string name) => Property(value, name) is { ValueKind: JsonValueKind.Number } number ? number.GetDouble() : null;
}
