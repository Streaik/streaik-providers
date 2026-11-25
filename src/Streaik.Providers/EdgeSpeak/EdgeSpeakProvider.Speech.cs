using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class EdgeSpeakProvider
{
    private const string SpeechPath = "audio/speech";

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.StreamFormat is not (null or "audio")) throw new ProviderRequestException("Unsupported EdgeSpeak speech stream_format.");
        var (body, package, buffered) = SpeechBody(request.Model, request.Input, request.Voice, request.ResponseFormat,
            request.Instructions, request.Speed, SpeechExtensions(request.AdditionalProperties), streaming: false);
        var mime = package ? "application/zip" : "audio/wav";
        var upstream = await SendMediaAsync(SpeechPath, JsonContent.Create(body, options: Json), mime, headers, ct, buffered);
        if (upstream.Content.Headers.ContentType?.MediaType == mime && upstream.Content.Headers.ContentLength != 0) return upstream;
        upstream.Dispose();
        throw new IOException($"EdgeSpeak did not return nonempty {mime} output.");
    }

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var (body, _, _) = SpeechBody(request.Model, request.Input, request.Voice, request.ResponseFormat,
            request.Instructions, request.Speed, SpeechExtensions(request.AdditionalProperties), streaming: true);
        return SpeechEvents(body, null, false, headers, ct);
    }

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Text is not { Length: > 0 } || request.Text.Any(part => part?.Type != "text" || part.Text is null))
            throw new ProviderRequestException("EdgeSpeak streaming speech requires text parts.");
        var options = ScopedOptions(request.ProviderOptions);
        if (request.Language is { } language) options["language"] = Value(language);
        var rate = request.OutputFormat?.Rate;
        if (rate is { } sampleRate)
        {
            if (sampleRate <= 0 || sampleRate > int.MaxValue || sampleRate != Math.Truncate(sampleRate))
                throw new ProviderRequestException("EdgeSpeak sample rate must be a positive integer.");
            options["sample_rate"] = Value((int)sampleRate);
        }
        if (request.Text.Any(part => part.Voice is not null || part.ProviderOptions?.ContainsKey(Id) == true))
        {
            if (options.ContainsKey("segments")) throw new ProviderRequestException("Use either speech part overrides or native segments.");
            var segments = new List<Dictionary<string, JsonElement>>();
            foreach (var part in request.Text.Where(part => part.Text.Length != 0))
            {
                var segment = ScopedOptions(part.ProviderOptions);
                if (segment.Keys.Any(key => key is not ("voice" or "instructions" or "disable_style" or "speed" or "language")))
                    throw new ProviderRequestException("Unsupported EdgeSpeak speech part override.");
                if (part.Voice is { } voice) segment["voice"] = Value(voice);
                if (segments.LastOrDefault() is { } last && last.Count == segment.Count + 1
                    && segment.All(p => last.TryGetValue(p.Key, out var value) && JsonElement.DeepEquals(p.Value, value)))
                    last["input"] = Value(last["input"].GetString() + part.Text);
                else
                {
                    segment["input"] = Value(part.Text);
                    segments.Add(segment);
                }
            }
            options["segments"] = Value(segments);
        }
        var (body, _, _) = SpeechBody(request.Model, string.Concat(request.Text.Select(part => part.Text)), request.Voice,
            request.OutputFormat?.Type, request.Instructions, request.Speed, options, streaming: true);
        return SpeechEvents(body, rate, request.IncludeRawChunks == true, headers, ct);
    }

    private async IAsyncEnumerable<StreamingSpeechPart> SpeechEvents(Dictionary<string, JsonElement> body, double? rate, bool includeRaw,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var upstream = await SendMediaAsync(SpeechPath, JsonContent.Create(body, options: Json), "text/event-stream", headers, ct);
        yield return new SpeechStreamStartPart();
        yield return new SpeechResponseMetadataPart { Timestamp = DateTime.UtcNow, ModelId = body["model"].GetString(), Headers = ResponseHeaders(upstream) };
        var receivedAudio = false;
        await foreach (var (type, data) in Events(upstream, "speech.audio.done", ct))
        {
            if (includeRaw) yield return new SpeechRawPart { RawValue = RawChunk(type, data) };
            if (type == "speech.audio.delta")
            {
                var audio = RequiredString(data, "audio");
                if (audio.Length == 0 || !System.Buffers.Text.Base64.IsValid(audio)) throw new JsonException("Invalid EdgeSpeak audio delta.");
                receivedAudio = true;
                yield return new SpeechAudioPart { Audio = audio, AudioFormat = new() { Type = "audio/wav", Rate = rate } };
            }
            else if (type == "speech.audio.done")
            {
                if (!receivedAudio) throw new IOException("EdgeSpeak completed speech without audio.");
                yield return new SpeechFinishPart { ProviderMetadata = Metadata(data) };
            }
        }
    }

    private static (Dictionary<string, JsonElement> Body, bool Package, bool Buffered) SpeechBody(string model, string? input, string? voice,
        string? format, string? instructions, double? speed, Dictionary<string, JsonElement> body, bool streaming)
    {
        var slash = model.LastIndexOf('/');
        if (slash > 0 && model[(slash + 1)..] is { } suffix && (suffix.StartsWith("builtin:", StringComparison.Ordinal) || suffix.StartsWith("user:", StringComparison.Ordinal)))
        {
            if (voice is not null && voice != suffix) throw new ProviderRequestException("Voice conflicts with the EdgeSpeak model's voice shortcut.");
            model = model[..slash];
            voice = suffix;
        }
        format ??= OptionString(body, "response_format");
        if (format?.ToLowerInvariant() is not (null or "wav" or "audio/wav" or "audio/x-wav"))
            throw new ProviderRequestException("EdgeSpeak speech supports WAV audio only.");
        var container = OptionString(body, "output_container") ?? "audio";
        if (container is not ("audio" or "zip")) throw new ProviderRequestException("EdgeSpeak output_container must be audio or zip.");
        var buffered = container == "zip" || body.TryGetValue("music", out var music) && music.ValueKind != JsonValueKind.Null;
        if (buffered && (streaming || model != "m-a-p/YuE2-3B"))
            throw new ProviderRequestException("EdgeSpeak music and ZIP output require buffered speech with m-a-p/YuE2-3B.");

        foreach (var key in body.Keys.Where(key => key.ToLowerInvariant() is "model" or "input" or "response_format" or "stream_format" or "stream").ToArray())
            body.Remove(key);
        body["model"] = Value(model);
        body["response_format"] = Value("wav");
        body["stream_format"] = Value(streaming ? "sse" : "audio");
        if (voice is not null) body["voice"] = Value(voice);
        if (string.IsNullOrWhiteSpace(OptionString(body, "voice"))) throw new ProviderRequestException("EdgeSpeak requires a voice ID.");
        if (instructions is not null) body["instructions"] = Value(instructions);
        if (speed is not null) body["speed"] = Value(speed);
        ValidateStyle(Value(body));
        if (body.TryGetValue("segments", out var segments))
        {
            if (segments.ValueKind != JsonValueKind.Array || segments.GetArrayLength() == 0)
                throw new ProviderRequestException("EdgeSpeak segments must be a non-empty array.");
            foreach (var segment in segments.EnumerateArray())
            {
                if (String(segment, "input") is null) throw new ProviderRequestException("Every EdgeSpeak speech segment requires input.");
                ValidateStyle(segment);
            }
        }
        else
        {
            if (string.IsNullOrWhiteSpace(input)) throw new ProviderRequestException("EdgeSpeak speech requires input.");
            body["input"] = Value(input);
        }
        return (body, container == "zip", buffered);
    }

    private static void ValidateStyle(JsonElement body)
    {
        if (Property(body, "disable_style").ValueKind == JsonValueKind.True && String(body, "instructions") is { Length: > 0 })
            throw new ProviderRequestException("EdgeSpeak disable_style conflicts with instructions.");
    }

    private static Dictionary<string, JsonElement> SpeechExtensions(Dictionary<string, JsonElement>? additional)
    {
        var result = new Dictionary<string, JsonElement>(additional ?? [], StringComparer.Ordinal);
        if (result.Remove("providerOptions", out var options))
            foreach (var (key, value) in ScopedOptions(Fields(options))) result[key] = value;
        if (result.Remove(Id, out var scoped))
            foreach (var (key, value) in Fields(scoped)) result[key] = value;
        return result;
    }

    private static string? OptionString(IReadOnlyDictionary<string, JsonElement> options, string name)
        => options.TryGetValue(name, out var value) && value.ValueKind != JsonValueKind.Null
            ? value.ValueKind == JsonValueKind.String ? value.GetString() : throw new ProviderRequestException($"EdgeSpeak {name} must be a string.") : null;
}
