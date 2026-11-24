using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class CartesiaProvider
{
    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var settings = SpeechPayload(request, streaming: false);
        using var client = clients.CreateClient(Id);
        using var outgoing = new HttpRequestMessage(HttpMethod.Post, "tts/bytes") { Content = JsonContent.Create(settings.Body, options: Json) };
        Auth(headers, settings.Version)(outgoing.Headers);
        NativeHttp.ApplyHeaders(outgoing, headers);
        return await NativeHttp.SendAsync(client, outgoing, ct);
    }

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamSpeechCoreAsync(request, SpeechPayload(request, streaming: true), false, headers, ct);

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Text is not { Length: > 0 } || request.Text.Any(part => part?.Text is null))
            throw new ProviderRequestException("Cartesia streaming speech requires text parts.");
        var audio = new AudioSpeechRequest { Model = request.Model, Voice = request.Text[0].Voice ?? request.Voice,
            Input = string.Concat(request.Text.Select(part => part.Text)), ResponseFormat = request.OutputFormat?.Type,
            Instructions = request.Instructions, Speed = request.Speed, StreamFormat = "sse" };
        SpeechSettings? first = null;
        foreach (var part in request.Text)
        {
            var options = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
            AddOptions(options, request.ProviderOptions);
            AddOptions(options, part.ProviderOptions);
            if (request.Language is { } lang) options["language"] = JsonSerializer.SerializeToElement(lang);
            if (request.OutputFormat?.Rate is { } rate) options["sampleRate"] = JsonSerializer.SerializeToElement(rate);
            audio.AdditionalProperties = options;
            audio.Voice = part.Voice ?? request.Voice;
            var current = SpeechPayload(audio, streaming: true);
            if (first is not null && (first.Version != current.Version || !JsonNode.DeepEquals(first.Body, current.Body)))
                throw new ProviderRequestException("Cartesia streaming speech requires one effective configuration across text parts.");
            first ??= current;
        }
        return StreamSpeechCoreAsync(audio, first!, request.IncludeRawChunks == true, headers, ct);
    }

    private static void AddOptions(Dictionary<string, JsonElement> target, Dictionary<string, JsonElement>? scoped)
    {
        if (scoped?.TryGetValue(Id, out var values) != true) return;
        if (values.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Cartesia speech options must be an object.");
        foreach (var option in values.EnumerateObject()) target[option.Name] = option.Value;
    }

    private async IAsyncEnumerable<StreamingSpeechPart> StreamSpeechCoreAsync(AudioSpeechRequest request, SpeechSettings settings, bool raw,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        using var outgoing = new HttpRequestMessage(HttpMethod.Post, "tts/sse") { Content = JsonContent.Create(settings.Body, options: Json) };
        Auth(headers, settings.Version)(outgoing.Headers);
        outgoing.Headers.Accept.ParseAdd("text/event-stream");
        NativeHttp.ApplyHeaders(outgoing, headers);
        using var response = await NativeHttp.SendAsync(client, outgoing, ct);
        if (response.Content.Headers.ContentType?.MediaType != "text/event-stream") throw new IOException("Expected Cartesia speech events.");
        yield return new SpeechStreamStartPart { Warnings = settings.Warnings };
        yield return new SpeechResponseMetadataPart { ModelId = request.Model, Timestamp = DateTime.UtcNow, Headers = SafeHeaders(response) };
        var format = new AudioFormat { Type = "audio/pcm", Rate = settings.Rate };
        var heardAudio = false;
        await foreach (var frame in NativeHttp.ReadSseAsync<JsonElement>(response, Json, part => String(part, "type") is "done" or "error", ct))
        {
            var type = String(frame, "type");
            if (type == "error") throw new IOException("Cartesia reported a speech streaming error.");
            if (raw) yield return new SpeechRawPart { RawValue = frame };
            var metadata = Metadata(type == "chunk" ? JsonSerializer.SerializeToElement(frame.EnumerateObject()
                .Where(field => field.Name != "data").ToDictionary(field => field.Name, field => field.Value)) : frame);
            if (type == "done")
            {
                if (!heardAudio) throw new IOException("Cartesia speech ended without audio.");
                yield return new SpeechFinishPart { ProviderMetadata = metadata };
                yield break;
            }
            var audio = type == "chunk" ? NormalizeSpeechAudio(String(frame, "data")) : "";
            heardAudio |= audio.Length > 0;
            yield return new SpeechAudioPart { Id = String(frame, "context_id"), Audio = audio, AudioFormat = format, ProviderMetadata = metadata };
        }
        throw new IOException("Cartesia speech ended without a done event.");
    }

    private static string NormalizeSpeechAudio(string? audio)
    {
        try { if (!string.IsNullOrWhiteSpace(audio)) return Convert.ToBase64String(Convert.FromBase64String(audio)); }
        catch (FormatException) { }
        throw new JsonException("Invalid Cartesia audio chunk.");
    }

    private static SpeechSettings SpeechPayload(AudioSpeechRequest request, bool streaming)
    {
        if (string.IsNullOrWhiteSpace(request.Input)) throw new ProviderRequestException("Cartesia requires nonempty speech input.");
        if (request.Model.Split('/') is not ["tts", var model, { Length: > 0 } voice] || !SpeechModels.Contains(model))
            throw new ProviderRequestException("Use a Cartesia tts/{model}/{voice} model.");

        var options = request.AdditionalProperties ?? [];
        var body = new JsonObject();
        foreach (var (name, value) in options)
            if (name is not ("apiVersion" or "container" or "encoding" or "sampleRate" or "bitRate" or "speed" or "volume" or "emotion" or "model" or "input" or "response_format" or "stream_format" or "stream"))
                body[name] = Copy(value);

        var warnings = new List<object>();
        if (!string.IsNullOrWhiteSpace(request.Instructions)) warnings.Add(new { type = "unsupported", feature = "instructions" });
        if (!string.IsNullOrWhiteSpace(request.Voice) && !request.Voice.Equals(voice, StringComparison.OrdinalIgnoreCase))
            warnings.Add(new { type = "ignored", feature = "voice", reason = "voice is derived from model id" });

        var output = body["output_format"] as JsonObject ?? new JsonObject();
        var nativeContainer = output["container"]?.ToString();
        var container = (request.ResponseFormat ?? Option(options, "container") ?? nativeContainer)?.Trim().ToLowerInvariant() switch
        {
            null => streaming ? "raw" : "wav",
            "wav" or "audio/wav" or "audio/x-wav" => "wav",
            "mp3" or "audio/mpeg" => "mp3",
            "raw" or "pcm" or "audio/pcm" or "application/octet-stream" => "raw",
            _ => throw new ProviderRequestException("Cartesia supports wav, mp3, and raw PCM speech output.")
        };
        if (streaming && container != "raw") throw new ProviderRequestException("Cartesia streaming speech requires raw PCM output.");
        output["container"] = container;
        output["sample_rate"] = CopyIfPresent(options, "sampleRate") ?? output["sample_rate"]?.DeepClone() ?? JsonValue.Create(44100);
        if (output["sample_rate"] is not JsonValue rateNode || !rateNode.TryGetValue<int>(out var rate) || rate <= 0)
            throw new ProviderRequestException("Cartesia sample rate must be a positive integer.");
        if (container == "mp3") output["bit_rate"] = CopyIfPresent(options, "bitRate") ?? output["bit_rate"]?.DeepClone() ?? JsonValue.Create(128000);
        else output["encoding"] = Option(options, "encoding") ?? output["encoding"]?.ToString() ?? "pcm_s16le";
        if (body["output_format"] != output) body["output_format"] = output;

        var config = body["generation_config"] as JsonObject ?? new JsonObject();
        var speed = request.Speed ?? NumberOption(options, "speed");
        var volume = NumberOption(options, "volume");
        if (speed is < 0.6 or > 1.5 || volume is < 0.5 or > 2.0)
            throw new ProviderRequestException("Cartesia requires speed 0.6–1.5 and volume 0.5–2.0.");
        if (model.StartsWith("sonic-3", StringComparison.Ordinal))
        {
            if (speed is { } s) config["speed"] = s;
            if (volume is { } v) config["volume"] = v;
            if (Option(options, "emotion") is { Length: > 0 } emotion) config["emotion"] = emotion;
        }
        if (config.Count > 0 && body["generation_config"] != config) body["generation_config"] = config;
        body["model_id"] = model;
        body["transcript"] = request.Input;
        body["voice"] = new JsonObject { ["mode"] = "id", ["id"] = voice };
        return new(body, Option(options, "apiVersion"), rate, warnings);
    }

    private static JsonNode? CopyIfPresent(Dictionary<string, JsonElement> options, string name) => options.TryGetValue(name, out var value) ? Copy(value) : null;
    private static double? NumberOption(Dictionary<string, JsonElement> options, string name) => options.TryGetValue(name, out var value)
        && value.ValueKind == JsonValueKind.Number && value.TryGetDouble(out var number) ? number : null;
    private sealed record SpeechSettings(JsonObject Body, string? Version, int Rate, List<object> Warnings);
}
