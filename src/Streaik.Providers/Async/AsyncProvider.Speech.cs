using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class AsyncProvider
{
    private sealed record SpeechPayload(JsonObject Body, string Format, string Model, string Voice);
    private static readonly HashSet<string> SpeechReserved = new(StringComparer.OrdinalIgnoreCase)
        { "model", "model_id", "input", "transcript", "instructions", "stream", "stream_format", "response_format" };
    private static readonly byte[] QuotaMarker = "--ERROR:QUOTA_EXCEEDED--"u8.ToArray();

    private static SpeechPayload Payload(AudioSpeechRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Input)) throw new ProviderRequestException("Speech input is required.");
        var (model, voice) = SpeechModel(request.Model);
        if (model is null) throw new ProviderRequestException($"Unsupported asyncAI speech model '{request.Model}'.");
        var body = Options(request.AdditionalProperties, SpeechReserved);
        if (body["voice"] is not JsonObject voiceOptions) body["voice"] = voiceOptions = new();
        voice ??= request.Voice?.Trim() ?? Text(voiceOptions, "id")?.Trim();
        if (string.IsNullOrWhiteSpace(voice)) throw new ProviderRequestException("An asyncAI voice ID is required.");
        voiceOptions["mode"] = "id";
        voiceOptions["id"] = voice;

        if (body["output_format"] is not JsonObject output) body["output_format"] = output = new();
        var format = (request.ResponseFormat ?? Text(output, "container") ?? "mp3").Trim().ToLowerInvariant() switch
        {
            "wav" => "wav", "pcm" or "pcm16" or "raw" => "raw", _ => "mp3"
        };
        output["container"] = format;
        output["sample_rate"] ??= 44100;
        if (format == "mp3") { output["bit_rate"] ??= 192000; output.Remove("encoding"); }
        else { output["encoding"] ??= "pcm_s16le"; output.Remove("bit_rate"); }
        body["model_id"] = model;
        body["transcript"] = request.Input;
        if (request.Speed is { } speed) body["speed_control"] = speed;
        return new(body, format, model, voice);

        static string? Text(JsonObject obj, string field) => obj[field] is JsonValue value && value.TryGetValue<string>(out var text) ? text : null;
    }

    private async Task<HttpResponseMessage> SendSpeechAsync(SpeechPayload payload, bool stream, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        var nativeStream = stream && payload.Format != "wav";
        var response = await SendAsync(client, nativeStream ? "text_to_speech/streaming" : "text_to_speech",
            JsonContent.Create(payload.Body, options: Json), headers, ct, nativeStream ? "application/octet-stream" : null);
        var contentType = response.Content.Headers.ContentType?.MediaType;
        if (contentType is null or "application/octet-stream" || contentType.StartsWith("audio/", StringComparison.OrdinalIgnoreCase))
        {
            response.Content.Headers.ContentType = new MediaTypeHeaderValue(payload.Format switch
            {
                "wav" => "audio/wav", "raw" => "application/octet-stream", _ => "audio/mpeg"
            });
            return response;
        }
        response.Dispose();
        throw new ProviderRequestException("asyncAI did not return audio.", 502);
    }

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var payload = Payload(request);
        var response = await SendSpeechAsync(payload, stream: false, headers, ct);
        var wrapped = new SpeechProviderResponse(response.Content,
            new() { [Id] = JsonSerializer.SerializeToElement(new { model = payload.Model, voice = payload.Voice }, Json) });
        response.Content = null;
        response.Dispose();
        return wrapped;
    }

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct) => StreamAudioAsync(Payload(request), headers, ct);

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Text is not { Length: > 0 } || request.Text.Any(part => part?.Text is null || part.ProviderOptions is not null)
            || request.Text.Select(part => part.Voice ?? request.Voice).Distinct().Count() != 1)
            throw new ProviderRequestException("asyncAI streaming supports one voice and no per-part options.");
        var payload = Payload(new AudioSpeechRequest { Model = request.Model, Input = string.Concat(request.Text.Select(part => part.Text)),
            Voice = request.Text[0].Voice ?? request.Voice, ResponseFormat = request.OutputFormat?.Type, Speed = request.Speed,
            AdditionalProperties = request.ProviderOptions?.GetValueOrDefault(Id) is { ValueKind: JsonValueKind.Object } options
                ? options.Deserialize<Dictionary<string, JsonElement>>(Json) : null });
        if (request.OutputFormat?.Rate is { } rate) payload.Body["output_format"]!["sample_rate"] = rate;
        return StreamAudioAsync(payload, headers, ct);
    }

    private async IAsyncEnumerable<StreamingSpeechPart> StreamAudioAsync(SpeechPayload payload,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var response = await SendSpeechAsync(payload, stream: true, headers, ct);
        await using var audio = await response.Content.ReadAsStreamAsync(ct);
        var buffer = new byte[16384 + QuotaMarker.Length - 1];
        var retained = 0;
        var start = false;
        while (await audio.ReadAsync(buffer.AsMemory(retained, 16384), ct) is > 0 and var count)
        {
            var size = count + retained;
            if (payload.Format != "wav" && buffer.AsSpan(0, size).IndexOf(QuotaMarker) >= 0)
                throw new ProviderRequestException("asyncAI speech quota exceeded during streaming.", 429);
            // Retain a possible partial marker across reads without buffering the rest of the audio.
            retained = payload.Format == "wav" ? 0 : MarkerSuffixLength(buffer, size);
            if (!start)
            {
                start = true;
                yield return new SpeechStreamStartPart();
                yield return new SpeechResponseMetadataPart { ModelId = payload.Model, Timestamp = DateTime.UtcNow };
            }
            if (size > retained) yield return AudioPart(size - retained);
            buffer.AsSpan(size - retained, retained).CopyTo(buffer);
        }
        if (!start) throw new ProviderRequestException("asyncAI returned an empty audio stream.", 502);
        if (retained > 0) yield return AudioPart(retained);
        yield return new SpeechFinishPart();

        SpeechAudioPart AudioPart(int length) => new()
        {
            Audio = Convert.ToBase64String(buffer, 0, length),
            AudioFormat = new() { Type = payload.Format }
        };
    }

    private static int MarkerSuffixLength(byte[] buffer, int count)
    {
        for (var length = Math.Min(count, QuotaMarker.Length - 1); length > 0; length--)
            if (buffer.AsSpan(count - length, length).SequenceEqual(QuotaMarker.AsSpan(0, length))) return length;
        return 0;
    }
}
