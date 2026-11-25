using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Speechmatics;

internal sealed partial class SpeechmaticsProvider
{
    private const int SampleRate = 16000;

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => await SendSpeechAsync(BuildSpeech(request), headers, ct);

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeechEvents(BuildSpeech(request), false, headers, ct);

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Text is not { Length: > 0 } || request.Text.Any(part => part is null || part.Type != "text" || part.Text is null))
            throw new ProviderRequestException("Speechmatics streaming speech requires text parts.");
        if (request.OutputFormat?.Rate is { } rate && rate != SampleRate)
            throw new ProviderRequestException("Speechmatics TTS supports a 16 kHz sample rate.", 422);
        var native = new AudioSpeechRequest { Model = request.Model, Input = string.Concat(request.Text.Select(part => part.Text)),
            Voice = request.Voice, ResponseFormat = request.OutputFormat?.Type, Instructions = request.Instructions,
            Speed = request.Speed, AdditionalProperties = request.ProviderOptions?.TryGetValue(Id, out var scoped) == true
                ? new() { [Id] = scoped } : null };
        var payload = BuildSpeech(native, request.Text.Select(part => part.Voice));
        if (!string.IsNullOrWhiteSpace(request.Language)) payload.Warnings.Add(Unsupported("language"));
        if (request.Text.Any(part => part.ProviderOptions?.Count > 0)) payload.Warnings.Add(Unsupported("per-part providerOptions"));
        return SpeechEvents(payload, request.IncludeRawChunks == true, headers, ct);
    }

    private async Task<SpeechProviderResponse> SendSpeechAsync(SpeechPayload speech, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var key = RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        var path = $"https://preview.tts.speechmatics.com/generate/{Uri.EscapeDataString(speech.Voice)}?output_format={speech.Format}_16000";
        using var upstream = await SendAsync(client, HttpMethod.Post, path, JsonContent.Create(speech.Body, options: Json), "Bearer", key, headers, ct);
        var content = upstream.Content;
        if (content.Headers.ContentType?.MediaType is { } type && type != "application/octet-stream"
            && !type.StartsWith("audio/", StringComparison.OrdinalIgnoreCase))
            throw new IOException("Speechmatics did not return audio.");
        var headersByName = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, values) in upstream.Headers.Concat(content.Headers)) headersByName[name] = string.Join(", ", values);
        var responseHeaders = NativeHttp.SafeResponseMetadataHeaders(headersByName);
        content.Headers.ContentType = new(speech.Format == "pcm" ? "application/octet-stream" : "audio/wav");
        var result = new SpeechProviderResponse(content, new() { [Id] = JsonSerializer.SerializeToElement(new { headers = responseHeaders }, Json) });
        upstream.Content = null; // Transfer ownership of the audio stream to the gateway.
        return result;
    }

    private async IAsyncEnumerable<StreamingSpeechPart> SpeechEvents(SpeechPayload speech, bool includeRaw,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var upstream = await SendSpeechAsync(speech, headers, ct);
        await using var audio = await upstream.Content.ReadAsStreamAsync(ct);
        var chunk = new byte[16384];
        var length = await audio.ReadAsync(chunk, ct);
        if (length == 0) throw new IOException("Speechmatics returned no speech audio.");
        yield return new SpeechStreamStartPart { Warnings = speech.Warnings };
        yield return new SpeechResponseMetadataPart { ModelId = speech.Model, Timestamp = DateTime.UtcNow };
        do
        {
            var encoded = Convert.ToBase64String(chunk, 0, length);
            if (includeRaw) yield return new SpeechRawPart { RawValue = encoded };
            yield return new SpeechAudioPart { Audio = encoded, AudioFormat = new() { Type = $"audio/{speech.Format}", Rate = SampleRate } };
        } while ((length = await audio.ReadAsync(chunk, ct)) > 0);
        yield return new SpeechFinishPart { ProviderMetadata = upstream.ProviderMetadata };
    }

    private SpeechPayload BuildSpeech(AudioSpeechRequest request, IEnumerable<string?>? partVoices = null)
    {
        if (!IsSpeechModel(request.Model) || string.IsNullOrWhiteSpace(request.Input))
            throw new ProviderRequestException("A Speechmatics TTS model and nonempty text are required.");
        var modelVoice = request.Model.StartsWith(VoiceSegment, StringComparison.Ordinal) ? request.Model[VoiceSegment.Length..] : null;
        var body = Options(request.AdditionalProperties, "model", "text", "input", "stream", "stream_format", "speed", "instructions", "language");
        var explicitVoice = request.Voice ?? ReadString(body, "voice");
        var voice = modelVoice ?? explicitVoice?.Trim();
        foreach (var partVoice in partVoices ?? [])
        {
            if (string.IsNullOrWhiteSpace(partVoice)) continue;
            if (voice is not null && !voice.Equals(partVoice.Trim(), StringComparison.OrdinalIgnoreCase))
                throw new ProviderRequestException("Speechmatics streaming speech supports one voice per request.", 422);
            voice ??= partVoice.Trim();
        }
        if (string.IsNullOrWhiteSpace(voice) || !voice.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_'))
            throw new ProviderRequestException("Select a valid Speechmatics voice with the voice field or a tts/voice model.");
        var format = (request.ResponseFormat ?? ReadString(body, "output_format") ?? ReadString(body, "response_format"))?.Trim().ToLowerInvariant() switch
        {
            null or "" or "wav" or "wave" or "wav_16000" or "audio/wav" or "audio/x-wav" => "wav",
            "pcm" or "pcm16" or "raw" or "pcm_16000" or "audio/pcm" or "audio/l16" => "pcm",
            _ => throw new ProviderRequestException("Speechmatics TTS supports WAV and PCM at 16 kHz.", 422)
        };
        body.Remove("voice");
        body.Remove("output_format");
        body.Remove("response_format");
        body["text"] = JsonSerializer.SerializeToElement(request.Input, Json);
        var warnings = new List<object>();
        if (modelVoice is not null && explicitVoice is not null && !modelVoice.Equals(explicitVoice.Trim(), StringComparison.OrdinalIgnoreCase))
            warnings.Add(new { type = "ignored", feature = "voice", reason = "voice is derived from the model ID" });
        if (!string.IsNullOrWhiteSpace(request.Instructions)) warnings.Add(Unsupported("instructions"));
        if (request.Speed is not null) warnings.Add(Unsupported("speed"));
        return new(request.Model, voice, format, body, warnings);
    }

    private static object Unsupported(string feature) => new { type = "unsupported", feature };
    private static string? ReadString(Dictionary<string, JsonElement> fields, string key)
        => fields.GetValueOrDefault(key) is { ValueKind: JsonValueKind.String } value ? value.GetString() : null;

    private sealed record SpeechPayload(string Model, string Voice, string Format, Dictionary<string, JsonElement> Body, List<object> Warnings);
}
