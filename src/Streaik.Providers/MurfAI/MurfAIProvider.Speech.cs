using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.Configuration;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class MurfAIProvider : IStreamingSpeechProvider, IStreamingSpeechEventsProvider
{
    private const string StreamUrl = "https://global.api.murf.ai/v1/speech/stream";
    private const int AudioLimit = 24 * 1024 * 1024;
    private static readonly HashSet<string> SpeechControls = new(StringComparer.OrdinalIgnoreCase)
    {
        "text", "input", "response_format", "stream", "stream_format", "instructions", "speed", "voice",
        "model", "modelVersion", "voiceId", "format", "language", "locale", "multiNativeLocale", "sampleRate", "rate"
    };
    private static readonly string[] BatchOnlyOptions = ["encodeAsBase64", "audioDuration", "channelType", "pitch", "variation", "wordDurationsAsOriginalText"];

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var speech = BuildSpeech(request.Model, request.Input, request.Voice, request.ResponseFormat, request.Speed, null, null,
            request.AdditionalProperties, streaming: false);
        return await RequestSpeechAsync(speech, ct);
    }

    public Task<HttpResponseMessage> SpeakStreamingAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => RequestSpeechAsync(BuildSpeech(request.Model, request.Input, request.Voice, request.ResponseFormat, request.Speed, null, null,
            request.AdditionalProperties, streaming: true), ct);

    private async Task<HttpResponseMessage> RequestSpeechAsync(SpeechCall speech, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        if (speech.IsFalcon)
        {
            var upstream = await SendAsync(client, StreamUrl, speech.Payload, ct);
            if (upstream.Content.Headers.ContentType?.MediaType is null or "application/octet-stream")
                upstream.Content.Headers.ContentType = new(Mime(speech.Format));
            return upstream;
        }

        using var response = await SendAsync(client, "speech/generate", speech.Payload, ct);
        var raw = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        var encoded = String(raw, "encodedAudio");
        if (string.IsNullOrWhiteSpace(encoded))
        {
            var location = String(raw, "audioFile");
            if (!Uri.TryCreate(location, UriKind.Absolute, out var uri) || uri.Scheme != Uri.UriSchemeHttps || !uri.IsDefaultPort
                || uri.UserInfo.Length != 0 || uri.Fragment.Length != 0 || uri.HostNameType != UriHostNameType.Dns || !IsAssetHost(uri.IdnHost))
                throw new JsonException("Missing or untrusted Murf AI audio URL.");
            using var request = new HttpRequestMessage(HttpMethod.Get, uri); // Never forward the API key to asset storage.
            using var download = await NativeHttp.SendAsync(client, request, ct);
            if (download.Content.Headers.ContentLength > AudioLimit) throw new JsonException("Oversized Murf AI audio download.");
            if (download.Content.Headers.ContentType?.MediaType is { } type && !type.StartsWith("audio/", StringComparison.OrdinalIgnoreCase)
                && type != "application/octet-stream") throw new JsonException("Invalid Murf AI audio content type.");
            await using var stream = await download.Content.ReadAsStreamAsync(ct);
            encoded = await BoundedBase64Async(stream, AudioLimit, ct);
        }
        if (encoded.Length > ((AudioLimit + 2) / 3) * 4) throw new JsonException("Oversized Murf AI audio result.");
        byte[] audio;
        try { audio = Convert.FromBase64String(encoded); }
        catch (FormatException ex) { throw new JsonException("Invalid Murf AI audio result.", ex); }
        if (audio.Length is 0 or > AudioLimit) throw new JsonException("Invalid Murf AI audio size.");
        var metadata = ToObject(raw);
        metadata.Remove("encodedAudio"); // The audio bytes are already returned by the contract.
        metadata.Remove("audioFile"); // Do not duplicate signed delivery URLs in provider metadata.
        var content = new ByteArrayContent(audio);
        content.Headers.ContentType = new(Mime(speech.Format));
        return new SpeechProviderResponse(content, NativeMediaOptions.Metadata(Id, metadata));
    }

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeechParts(BuildSpeech(request.Model, request.Input, request.Voice, request.ResponseFormat, request.Speed, null, null,
            request.AdditionalProperties, streaming: true), request.Model, request.Instructions, rawChunks: false, ct);

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Text is not { Length: > 0 }) throw new ProviderRequestException("Murf AI streaming requires at least one text part.");
        SpeechCall? speech = null;
        foreach (var part in request.Text)
        {
            if (part is null || part.Type != "text" || part.Text is null) throw new ProviderRequestException("Murf AI speech parts must contain text.");
            var options = Scoped(request.ProviderOptions);
            foreach (var (key, value) in Scoped(part.ProviderOptions)) options[key] = value;
            var current = BuildSpeech(request.Model, "validation", part.Voice ?? request.Voice, request.OutputFormat?.Type, request.Speed,
                request.Language, request.OutputFormat?.Rate, options, streaming: true);
            if (speech is not null && !JsonNode.DeepEquals(speech.Payload, current.Payload))
                throw new ProviderRequestException("Murf AI HTTP streaming requires one voice and configuration across text parts.", 422);
            speech ??= current;
        }
        var text = string.Concat(request.Text.Select(part => part.Text));
        if (string.IsNullOrWhiteSpace(text)) throw new ProviderRequestException("Murf AI speech text cannot be empty.");
        speech!.Payload["text"] = text;
        return SpeechParts(speech, request.Model, request.Instructions, request.IncludeRawChunks == true, ct);
    }

    private async IAsyncEnumerable<StreamingSpeechPart> SpeechParts(SpeechCall speech, string model, string? instructions, bool rawChunks,
        [EnumeratorCancellation] CancellationToken ct)
    {
        using var upstream = await RequestSpeechAsync(speech, ct);
        yield return new SpeechStreamStartPart { Warnings = string.IsNullOrWhiteSpace(instructions) ? [] : [new { type = "unsupported", feature = "instructions" }] };
        yield return new SpeechResponseMetadataPart { ModelId = model, Timestamp = DateTime.UtcNow };
        var rate = speech.SampleRate ?? 24000;
        var metadata = NativeMediaOptions.Metadata(Id, new { format = speech.Format.ToUpperInvariant(), sample_rate = rate });
        await using var stream = await upstream.Content.ReadAsStreamAsync(ct);
        var buffer = new byte[16 * 1024];
        var hasAudio = false;
        while (await stream.ReadAsync(buffer, ct) is > 0 and var length)
        {
            hasAudio = true;
            var audio = Convert.ToBase64String(buffer, 0, length);
            if (rawChunks) yield return new SpeechRawPart { RawValue = audio };
            yield return new SpeechAudioPart { Audio = audio, AudioFormat = new() { Type = Mime(speech.Format), Rate = rate }, ProviderMetadata = metadata };
        }
        if (!hasAudio) throw new JsonException("Murf AI returned no audio in the stream.");
        yield return new SpeechFinishPart { ProviderMetadata = metadata };
    }

    private static SpeechCall BuildSpeech(string model, string? text, string? voice, string? format, double? speed, string? language,
        double? sampleRate, IReadOnlyDictionary<string, JsonElement>? extras, bool streaming)
    {
        if (string.IsNullOrWhiteSpace(text)) throw new ProviderRequestException("Murf AI speech text cannot be empty.");
        var options = new Dictionary<string, JsonElement>(NativeMediaOptions.Read(Id, extras, new HashSet<string>(StringComparer.OrdinalIgnoreCase)), StringComparer.OrdinalIgnoreCase);
        var parts = model.Split('/', StringSplitOptions.TrimEntries);
        if (parts.Length > 2 || parts.Any(string.IsNullOrWhiteSpace)) throw new ProviderRequestException("Invalid Murf AI speech model.");
        var version = NormalizeVersion(parts[0]);
        if (version is not ("gen2" or Falcon)) version = NormalizeVersion(OptionString(options, "modelVersion") ?? OptionString(options, "model") ?? (streaming ? Falcon : "gen2"));
        if (version is not ("gen2" or Falcon) || (streaming && version != Falcon))
            throw new ProviderRequestException("Murf AI supports GEN2 batch speech and Falcon 2.1 streaming speech.", 422);
        var falcon = version == Falcon;
        voice = parts.Length == 2 ? parts[1] : (streaming ? voice ?? OptionString(options, "voiceId") : OptionString(options, "voiceId") ?? voice);
        if (string.IsNullOrWhiteSpace(voice)) throw new ProviderRequestException("Murf AI requires a voice ID.");
        format = NormalizeFormat(format ?? OptionString(options, "format") ?? (streaming ? "mp3" : "wav"));
        language = falcon ? language ?? OptionString(options, "locale") ?? OptionString(options, "multiNativeLocale") ?? OptionString(options, "language")
            : OptionString(options, "multiNativeLocale") ?? language ?? OptionString(options, "language");
        sampleRate ??= ReadInt(options, "sampleRate");
        if (sampleRate is <= 0) throw new ProviderRequestException("Murf AI sample rate must be positive.");
        var rate = speed is { } s ? (int)Math.Clamp(Math.Round((s - 1.0) * 50.0, MidpointRounding.AwayFromZero), -50, 50) : ReadInt(options, "rate");
        var payload = ToObject(options.Where(pair => !SpeechControls.Contains(pair.Key)
            && (!falcon || !BatchOnlyOptions.Contains(pair.Key, StringComparer.OrdinalIgnoreCase))).ToDictionary());
        payload["text"] = text;
        payload["voiceId"] = voice.Trim();
        payload[falcon ? "model" : "modelVersion"] = falcon ? Falcon : "GEN2";
        payload["format"] = format.ToUpperInvariant();
        if (!string.IsNullOrWhiteSpace(language)) payload[falcon ? "locale" : "multiNativeLocale"] = language.Trim();
        if (rate is { } r) payload["rate"] = Math.Clamp(r, -50, 50);
        if (sampleRate is { } hz) payload["sampleRate"] = hz;
        if (!falcon)
        {
            payload["encodeAsBase64"] = !options.TryGetValue("encodeAsBase64", out var encode) || encode.ValueKind != JsonValueKind.False;
            if (ReadInt(options, "pitch") is { } pitch) payload["pitch"] = Math.Clamp(pitch, -50, 50);
            if (ReadInt(options, "variation") is { } variation) payload["variation"] = Math.Clamp(variation, 0, 5);
        }
        return new(payload, format, falcon, sampleRate);
    }

    private static Dictionary<string, JsonElement> Scoped(Dictionary<string, JsonElement>? extras)
        => NativeMediaOptions.Read(Id, extras?.TryGetValue(Id, out var value) == true ? new Dictionary<string, JsonElement> { [Id] = value } : null,
            new HashSet<string>(StringComparer.OrdinalIgnoreCase));

    private static int? ReadInt(Dictionary<string, JsonElement> options, string key)
        => options.TryGetValue(key, out var value) && value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out var number) ? number : null;

    private bool IsAssetHost(string host) => host.EndsWith(".murf.ai", StringComparison.OrdinalIgnoreCase)
        || configuration.GetSection("Providers:murfai:AssetHosts").GetChildren().Any(item => string.Equals(host, item.Value, StringComparison.OrdinalIgnoreCase));

    private static string NormalizeVersion(string version) => version.Trim().ToLowerInvariant() is "falcon-2" ? Falcon : version.Trim().ToLowerInvariant();
    private static string NormalizeFormat(string format) => format.Trim().ToLowerInvariant() switch
    {
        "mpeg" or "audio/mpeg" => "mp3", "wave" or "audio/wav" or "audio/wave" => "wav",
        "audio/flac" => "flac", "audio/ogg" => "ogg", var value => value
    };

    private static string Mime(string format) => format switch
    {
        "mp3" => "audio/mpeg", "wav" => "audio/wav", "flac" => "audio/flac", "ogg" => "audio/ogg",
        "alaw" => "audio/alaw", "ulaw" => "audio/basic", "pcm" => "audio/pcm", _ => "application/octet-stream"
    };

    private sealed record SpeechCall(JsonObject Payload, string Format, bool IsFalcon, double? SampleRate);
}
