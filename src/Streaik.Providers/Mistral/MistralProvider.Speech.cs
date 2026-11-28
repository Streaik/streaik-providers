using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Mistral;

internal sealed partial class MistralProvider
{
    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        var (body, _, format) = await SpeechBodyAsync(request, AudioOptions(request.AdditionalProperties), false, [], ct);
        var result = await NativeHttp.PostAsync<Dictionary<string, object?>, JsonElement>(client, "audio/speech", body,
            Bearer(ApiKey()), headers, Json, ct);
        var mime = SpeechMime(SpeechFormat(String(result, "audio_format") ?? format, upstream: true));
        var content = new ByteArrayContent(AudioBytes(RequiredString(result, "audio_data"), upstream: true));
        content.Headers.ContentType = new MediaTypeHeaderValue(mime);
        var metadata = result.EnumerateObject().Where(option => option.Name != "audio_data").ToDictionary(x => x.Name, x => x.Value);
        return new SpeechProviderResponse(content, AudioMetadata(JsonSerializer.SerializeToElement(metadata)));
    }

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeechEventsAsync(request, AudioOptions(request.AdditionalProperties), headers, false, [], ct);

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Text is not { Length: > 0 } || request.Text.Any(part => part is null || part.Type != "text" || part.Text is null))
            throw new ProviderRequestException("Mistral speech needs at least one text part.");
        var defaults = AudioOptions(request.ProviderOptions, namespaced: true);
        (Dictionary<string, JsonElement> Options, string? Voice, string Format) Effective(StreamingSpeechTextPart part)
        {
            var options = new Dictionary<string, JsonElement>(defaults, StringComparer.Ordinal);
            foreach (var field in AudioOptions(part.ProviderOptions, namespaced: true)) options[field.Key] = field.Value;
            var voice = ReadOption(options, "voice_id") ?? part.Voice ?? request.Voice;
            var format = SpeechFormat(request.OutputFormat?.Type ?? ReadOption(options, "response_format"));
            options.Remove("voice_id");
            options.Remove("response_format");
            return (options, voice, format);
        }
        var first = Effective(request.Text[0]);
        foreach (var part in request.Text.Skip(1))
        {
            var next = Effective(part);
            if (!string.Equals(first.Voice, next.Voice, StringComparison.OrdinalIgnoreCase) || first.Format != next.Format
                || !JsonElement.DeepEquals(JsonSerializer.SerializeToElement(first.Options), JsonSerializer.SerializeToElement(next.Options)))
                throw new ProviderRequestException("Mistral speech requires the same voice and options for each text part.");
        }
        var warnings = new List<object>();
        if (request.OutputFormat?.Rate is <= 0) throw new ProviderRequestException("Speech output sample rate must be positive.");
        if (request.OutputFormat?.Rate is not null) warnings.Add(new { type = "unsupported", feature = "outputFormat.rate" });
        if (request.Language is not null) warnings.Add(new { type = "unsupported", feature = "language" });
        return SpeechEventsAsync(new() { Model = request.Model, Input = string.Concat(request.Text.Select(part => part.Text)),
            Voice = first.Voice, ResponseFormat = first.Format, Instructions = request.Instructions, Speed = request.Speed },
            first.Options, headers, request.IncludeRawChunks == true, warnings, ct);
    }

    private async IAsyncEnumerable<StreamingSpeechPart> SpeechEventsAsync(AudioSpeechRequest request, Dictionary<string, JsonElement> options,
        IReadOnlyDictionary<string, string> headers, bool raw, List<object> warnings, [EnumeratorCancellation] CancellationToken ct)
    {
        var (body, model, format) = await SpeechBodyAsync(request, options, true, warnings, ct);
        using var client = CreateClient();
        using var message = new HttpRequestMessage(HttpMethod.Post, "audio/speech") { Content = JsonContent.Create(body) };
        using var response = await SendAudioAsync(client, message, headers, "text/event-stream", ct);
        yield return new SpeechStreamStartPart { Warnings = warnings };
        yield return new SpeechResponseMetadataPart { Timestamp = DateTime.UtcNow, ModelId = model };
        await foreach (var (type, payload) in AudioEventsAsync(response, "speech.audio.done", ct))
        {
            if (raw) yield return new SpeechRawPart { RawValue = payload };
            if (type == "speech.audio.delta")
            {
                var audio = RequiredString(payload, "audio_data");
                _ = AudioBytes(audio, upstream: true);
                yield return new SpeechAudioPart { Audio = audio, AudioFormat = new() { Type = SpeechMime(format) } };
            }
            else if (type == "speech.audio.done")
            {
                var usage = Property(payload, "usage");
                yield return new SpeechFinishPart { ProviderMetadata = AudioMetadata(payload), Usage = usage.ValueKind == JsonValueKind.Object
                    ? new() { InputCharacters = AudioCount(usage, "input_characters"),
                        InputTokens = AudioCount(usage, "input_tokens") ?? AudioCount(usage, "prompt_tokens"),
                        OutputTokens = AudioCount(usage, "output_tokens") ?? AudioCount(usage, "completion_tokens"),
                        Raw = usage.EnumerateObject().ToDictionary(x => x.Name, x => x.Value) } : null };
            }
        }
    }

    private async Task<(Dictionary<string, object?> Body, string Model, string Format)> SpeechBodyAsync(AudioSpeechRequest request,
        Dictionary<string, JsonElement> options, bool stream, List<object> warnings, CancellationToken ct)
    {
        if (!IsSpeechModel(request.Model) || string.IsNullOrWhiteSpace(request.Input))
            throw new ProviderRequestException("Mistral speech requires a Voxtral TTS model and nonempty text.");
        var parts = request.Model.Split('/', 2);
        var voice = (ReadOption(options, "voice_id") ?? request.Voice)?.Trim();
        if (parts.Length == 2 && string.IsNullOrWhiteSpace(parts[1])) throw new ProviderRequestException("Missing Mistral voice slug.");
        if (parts.Length == 2 || !string.IsNullOrWhiteSpace(voice))
        {
            var selected = parts.Length == 2 ? parts[1] : voice!;
            if (parts.Length == 2 && voice is { Length: > 0 } && !string.Equals(voice, selected, StringComparison.OrdinalIgnoreCase))
                warnings.Add(new { type = "ignored", feature = "voice", reason = "voice is derived from the model ID" });
            var match = await FindVoiceAsync(selected, ct);
            if (parts.Length == 2 && match is null) throw new ProviderRequestException("The Mistral voice shortcut is not available.");
            voice = match?.Id ?? selected;
        }
        if (request.Instructions is not null) warnings.Add(new { type = "unsupported", feature = "instructions" });
        if (request.Speed is not null) warnings.Add(new { type = "unsupported", feature = "speed" });
        var format = SpeechFormat(request.ResponseFormat ?? ReadOption(options, "response_format"));
        var body = new Dictionary<string, object?>(StringComparer.Ordinal);
        if (options.TryGetValue("voice_settings", out var settings)) body["voice_settings"] = settings;
        body["input"] = request.Input;
        body["model"] = parts[0];
        body["stream"] = stream;
        body["response_format"] = format;
        if (!string.IsNullOrWhiteSpace(voice)) body["voice_id"] = voice;
        return (body, parts[0], format);
    }

    private static string? ReadOption(Dictionary<string, JsonElement> options, string name)
        => options.GetValueOrDefault(name) switch
        {
            { ValueKind: JsonValueKind.String } value => value.GetString(),
            { ValueKind: JsonValueKind.Null or JsonValueKind.Undefined } => null,
            _ => throw new ProviderRequestException($"Mistral speech option '{name}' must be a string.")
        };

    private static string SpeechFormat(string? format, bool upstream = false) => format?.Trim().ToLowerInvariant() switch
    {
        null or "" or "mp3" or "audio/mp3" or "audio/mpeg" => "mp3",
        "wav" or "wave" or "audio/wav" or "audio/x-wav" => "wav",
        "pcm" or "pcm_s16le" or "audio/pcm" or "audio/l16" => "pcm",
        "flac" or "audio/flac" => "flac",
        "opus" or "audio/ogg" or "audio/opus" => "opus",
        _ when upstream => throw new JsonException("Mistral returned an unrecognized audio format."),
        _ => throw new ProviderRequestException("Unsupported Mistral speech output format.")
    };

    private static string SpeechMime(string format) => format switch
    { "mp3" => "audio/mpeg", _ => "audio/" + format };
}
