using System.Net.Http.Json;
using System.Net.ServerSentEvents;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class NeuphonicProviderRegistration
{
    public static IServiceCollection AddNeuphonic(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, NeuphonicProvider.Id, "https://api.neuphonic.com/");
        return services.AddProvider<NeuphonicProvider>(NeuphonicProvider.Id);
    }
}

internal sealed class NeuphonicProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, ISpeechProvider, IStreamingSpeechEventsProvider, IChatAudioRoutingProvider
{
    internal const string Id = "neuphonic";
    private const int AudioLimit = 25 * 1024 * 1024;
    private static readonly HashSet<string> Reserved = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "text", "input", "voice", "voice_id", "response_format", "stream_format", "stream", "instructions"
    };

    public bool IsSpeechModel(string model) => model?.IndexOf('/') is > 0 and var separator && separator < model.Length - 1;
    public bool IsTranscriptionModel(string model) => false;

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var key = credentials.GetApiKey(Id);
        if (string.IsNullOrWhiteSpace(key)) yield break;
        using var client = clients.CreateClient(Id);
        var body = await NativeHttp.GetAsync<JsonElement>(client, "voices", h => h.Add("x-api-key", key), ct);
        var voices = Property(Property(body, "data"), "voices");
        if (voices.ValueKind != JsonValueKind.Array) throw new JsonException("Expected a Neuphonic data.voices array.");
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var voice in voices.EnumerateArray())
        {
            var language = String(voice, "lang_code");
            var id = String(voice, "voice_id");
            if (string.IsNullOrWhiteSpace(language) || string.IsNullOrWhiteSpace(id)) continue;
            var model = $"{language}/{id}";
            if (seen.Add(model)) yield return new() { Id = model, Name = String(voice, "name") ?? id, OwnedBy = Id, Type = "speech" };
        }
    }

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var config = Build(request.Model, request.Voice, request.ResponseFormat, request.Instructions, request.Speed,
            NativeMediaOptions.Read(Id, request.AdditionalProperties, Reserved));
        var parts = StreamAsync(config, request.Input, raw: false, ct);
        using var audio = new MemoryStream();
        Dictionary<string, JsonElement>? metadata = null;
        await foreach (var part in parts)
        {
            if (part is SpeechAudioPart chunk)
            {
                var bytes = Convert.FromBase64String(chunk.Audio);
                if (audio.Length + bytes.Length > AudioLimit) throw new ProviderRequestException("Neuphonic audio exceeds the inline size limit.", 413);
                audio.Write(bytes);
            }
            else if (part is SpeechFinishPart finish) metadata = finish.ProviderMetadata;
        }
        var content = new ByteArrayContent(audio.ToArray());
        content.Headers.ContentType = new(config.MediaType);
        return new SpeechProviderResponse(content, metadata);
    }

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamAsync(Build(request.Model, request.Voice, request.ResponseFormat, request.Instructions, request.Speed,
            NativeMediaOptions.Read(Id, request.AdditionalProperties, Reserved)), request.Input, raw: false, ct);

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Text is not { Length: > 0 }) throw new ProviderRequestException("Neuphonic speech requires text parts.");
        if (request.OutputFormat?.Rate is <= 0 or > int.MaxValue)
            throw new ProviderRequestException("Neuphonic sampling rate must be a positive integer.");

        var defaults = ScopedOptions(request.ProviderOptions);
        Config? config = null;
        var text = new StringBuilder();
        foreach (var part in request.Text)
        {
            if (part?.Text is null || part.Type != "text") throw new ProviderRequestException("Neuphonic supports text parts only.");
            var options = new Dictionary<string, JsonElement>(defaults, StringComparer.Ordinal);
            foreach (var (name, value) in ScopedOptions(part.ProviderOptions)) options[name] = value;
            if (request.Language is { } language) options["language"] = JsonSerializer.SerializeToElement(language);
            if (request.OutputFormat?.Rate is { } rate)
            {
                if (rate != (int)rate) throw new ProviderRequestException("Neuphonic sampling rate must be a positive integer.");
                options["sampling_rate"] = JsonSerializer.SerializeToElement((int)rate);
            }
            var current = Build(request.Model, part.Voice ?? request.Voice, request.OutputFormat?.Type, request.Instructions, request.Speed, options);
            if (config is not null && !JsonObject.DeepEquals(config.Payload, current.Payload))
                throw new ProviderRequestException("Neuphonic streaming requires one effective voice and configuration across text parts.", 422);
            config = current;
            text.Append(part.Text);
        }
        return StreamAsync(config!, text.ToString(), request.IncludeRawChunks == true, ct);
    }

    private async IAsyncEnumerable<StreamingSpeechPart> StreamAsync(Config config, string text, bool raw, [EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(text)) throw new ProviderRequestException("Neuphonic speech requires text.");
        config.Payload["text"] = text;
        using var client = clients.CreateClient(Id);
        using var request = new HttpRequestMessage(HttpMethod.Post, $"sse/speak/{Uri.EscapeDataString(config.Language)}")
        {
            Content = JsonContent.Create(config.Payload, options: Json)
        };
        request.Headers.Add("x-api-key", RequiredKey(credentials, Id));
        request.Headers.Accept.ParseAdd("text/event-stream");
        using var response = await NativeHttp.SendAsync(client, request, ct);
        if (response.Content.Headers.ContentType?.MediaType != "text/event-stream")
            throw new JsonException("Expected a Neuphonic speech event stream.");
        var rawHeaders = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, values) in response.Headers.Concat(response.Content.Headers)) rawHeaders[name] = string.Join(", ", values);
        var headers = NativeHttp.SafeResponseMetadataHeaders(rawHeaders);
        var events = new List<JsonElement>();
        var started = false;
        var receivedAudio = false;
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        await foreach (var item in SseParser.Create(stream).EnumerateAsync(ct))
        {
            if (string.IsNullOrWhiteSpace(item.Data)) continue;
            if (item.Data == "[DONE]") break;
            using var document = JsonDocument.Parse(item.Data);
            var root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object) throw new JsonException("Expected a Neuphonic speech event object.");
            var status = Property(root, "status_code");
            int? code = status.ValueKind switch
            {
                JsonValueKind.Undefined or JsonValueKind.Null => null,
                JsonValueKind.Number when status.TryGetInt32(out var number) => number,
                _ => throw new JsonException("Expected a numeric Neuphonic status_code.")
            };
            var errors = Property(root, "errors");
            if (item.EventType == "error" || code is < 200 or >= 300
                || errors.ValueKind is not (JsonValueKind.Undefined or JsonValueKind.Null) && (errors.ValueKind != JsonValueKind.Array || errors.GetArrayLength() != 0))
                throw new HttpRequestException("Neuphonic speech failed.", null,
                    code is >= 400 and <= 599 ? (System.Net.HttpStatusCode)code.Value : null);

            var data = Property(root, "data");
            var audio = Property(data, "audio");
            string? base64 = null;
            if (audio.ValueKind is not (JsonValueKind.Undefined or JsonValueKind.Null))
            {
                if (audio.ValueKind != JsonValueKind.String || !System.Buffers.Text.Base64.IsValid(audio.GetString(), out var bytes) || bytes > AudioLimit)
                    throw new JsonException("Expected valid Neuphonic base64 audio.");
                if (bytes > 0) base64 = audio.GetString();
            }

            var metadata = ToObject(root);
            if (metadata["data"] is JsonObject details) details.Remove("audio");
            var frame = JsonSerializer.SerializeToElement(metadata, Json);
            events.Add(frame);
            if (!started)
            {
                started = true;
                yield return new SpeechStreamStartPart();
                yield return new SpeechResponseMetadataPart { Timestamp = DateTime.UtcNow, ModelId = $"{config.Language}/{config.Voice}",
                    Headers = headers.Where(pair => pair.Value is not null).ToDictionary(pair => pair.Key, pair => pair.Value!) };
            }
            if (raw) yield return new SpeechRawPart { RawValue = frame };
            if (base64 is null) continue;
            receivedAudio = true;
            var rate = Property(data, "sampling_rate");
            yield return new SpeechAudioPart
            {
                Audio = base64,
                AudioFormat = new() { Type = config.MediaType, Rate = rate.ValueKind == JsonValueKind.Number && rate.TryGetInt32(out var sampleRate)
                    && sampleRate > 0 ? sampleRate : config.Rate },
                ProviderMetadata = new() { [Id] = frame }
            };
        }
        if (!receivedAudio) throw new JsonException("Neuphonic returned no speech audio.");
        yield return new SpeechFinishPart { ProviderMetadata = NativeMediaOptions.Metadata(Id, new { events, headers, audioFormat = config.Format }) };
    }

    private static Dictionary<string, JsonElement> ScopedOptions(Dictionary<string, JsonElement>? options)
        => options?.TryGetValue(Id, out var native) == true
            ? NativeMediaOptions.Read(Id, new Dictionary<string, JsonElement> { [Id] = native }, Reserved) : new(StringComparer.Ordinal);

    private Config Build(string model, string? voice, string? format, string? instructions, double? speed, Dictionary<string, JsonElement> options)
    {
        if (!IsSpeechModel(model)) throw new ProviderRequestException("Neuphonic models must use lang_code/voice_id.");
        if (instructions is not null) throw new ProviderRequestException("Neuphonic does not support instructions.", 422);
        if (format is not (null or "pcm" or "raw" or "audio/pcm")) throw new ProviderRequestException("Neuphonic supports raw PCM audio only.", 422);
        if (options.Remove("output_format", out var output) && output.ValueKind != JsonValueKind.Null)
            throw new ProviderRequestException("Neuphonic does not accept a native output_format field.", 422);

        var separator = model.IndexOf('/');
        var language = TextOption("language") ?? TextOption("lang_code") ?? model[..separator];
        voice ??= model[(separator + 1)..];
        if (string.IsNullOrWhiteSpace(language) || language.Any(c => !char.IsAsciiLetterOrDigit(c) && c is not ('-' or '_'))
            || string.IsNullOrWhiteSpace(voice)) throw new ProviderRequestException("A valid Neuphonic language and voice are required.");
        options.Remove("language");
        var encoding = TextOption("encoding");
        if (encoding is not (null or "pcm_linear" or "pcm_mulaw"))
            throw new ProviderRequestException("Neuphonic supports pcm_linear and pcm_mulaw encodings only.", 422);
        if (format is not null) encoding = "pcm_linear";
        options.Remove("encoding");
        int? rate = null;
        if (options.TryGetValue("sampling_rate", out var configuredRate) && configuredRate.ValueKind == JsonValueKind.Number
            && configuredRate.TryGetInt32(out var number) && number > 0) rate = number;

        var payload = new JsonObject();
        foreach (var (name, value) in options) payload[name] = Copy(value);
        payload["lang_code"] = language;
        payload["voice_id"] = voice;
        if (encoding is not null) payload["encoding"] = encoding;
        if (speed is { } valueSpeed && double.IsFinite(valueSpeed) && valueSpeed > 0) payload["speed"] = valueSpeed;
        else if (speed is not null) throw new ProviderRequestException("Neuphonic speed must be positive.");
        return new(payload, language, voice, encoding == "pcm_mulaw" ? "application/octet-stream" : "audio/pcm",
            encoding == "pcm_mulaw" ? "pcm_mulaw" : "pcm", rate);

        string? TextOption(string name)
            => !options.TryGetValue(name, out var value) || value.ValueKind == JsonValueKind.Null ? null
                : value.ValueKind == JsonValueKind.String ? value.GetString() : throw new ProviderRequestException($"Neuphonic {name} must be a string.");
    }

    private sealed record Config(JsonObject Payload, string Language, string Voice, string MediaType, string Format, int? Rate);
}
