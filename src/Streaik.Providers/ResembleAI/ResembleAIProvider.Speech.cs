using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class ResembleAIProvider
{
    private const string Synthesis = "https://f.cluster.resemble.ai/";
    private static readonly HashSet<string> SpeechReserved = new(StringComparer.OrdinalIgnoreCase)
        { "model", "input", "data", "voice", "stream", "stream_format", "response_format", "instructions", "speed" };

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var payload = SpeechPayload(request.Model, request.Input, request.Voice, request.ResponseFormat,
            NativeMediaOptions.Read(Id, request.AdditionalProperties, SpeechReserved), null, streaming: false);
        using var client = clients.CreateClient(Id);
        var (root, upstream) = await NativeHttp.PostJsonAsync(client, Synthesis + "synthesize", payload, Auth(), headers, Json, ct);
        if (Property(root, "success").ValueKind == JsonValueKind.False || String(root, "audio_content") is not { Length: > 0 } encoded)
            throw new JsonException("ResembleAI did not return audio content.");
        byte[] bytes;
        try { bytes = Convert.FromBase64String(encoded); }
        catch (FormatException e) { throw new JsonException("ResembleAI returned invalid audio content.", e); }
        if (bytes.Length == 0) throw new JsonException("ResembleAI returned empty audio content.");
        var format = NormalizeFormat(String(root, "output_format") ?? OptionString(payload, "output_format"));
        var audio = new ByteArrayContent(bytes);
        audio.Headers.ContentType = new(format == "mp3" ? "audio/mpeg" : "audio/wav");
        var metadata = root.EnumerateObject().Where(p => p.Name != "audio_content").ToDictionary(p => p.Name, p => p.Value);
        var response = new SpeechProviderResponse(audio, NativeMediaOptions.Metadata(Id, metadata));
        foreach (var (name, value) in NativeHttp.SafeResponseMetadataHeaders(upstream))
            response.Headers.TryAddWithoutValidation(name, value);
        return response;
    }

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamSpeechCoreAsync(request.Model, SpeechPayload(request.Model, request.Input, request.Voice, request.ResponseFormat,
            NativeMediaOptions.Read(Id, request.AdditionalProperties, SpeechReserved), null, streaming: true),
            request.Instructions, false, headers, ct);

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Text is not { Length: > 0 } || request.OutputFormat?.Rate is <= 0)
            throw new ProviderRequestException("ResembleAI requires speech text and a positive sample rate when supplied.");
        var defaults = SpeechOptions(request.ProviderOptions);
        var text = new StringBuilder();
        Dictionary<string, JsonElement>? payload = null;
        JsonElement? configuration = null;
        foreach (var part in request.Text)
        {
            if (part is not { Type: "text", Text: not null }) throw new ProviderRequestException("ResembleAI speech parts must contain text.");
            var options = new Dictionary<string, JsonElement>(defaults, StringComparer.OrdinalIgnoreCase);
            foreach (var (name, value) in SpeechOptions(part.ProviderOptions)) options[name] = value;
            var current = SpeechPayload(request.Model, "validation", part.Voice ?? request.Voice,
                request.OutputFormat?.Type, options, request.OutputFormat?.Rate, streaming: true);
            var serialized = JsonSerializer.SerializeToElement(current);
            if (configuration is { } previous && !JsonElement.DeepEquals(previous, serialized))
                throw new ProviderRequestException("ResembleAI requires one voice and provider configuration per stream.");
            configuration = serialized;
            payload = current;
            text.Append(part.Text);
        }
        if (string.IsNullOrWhiteSpace(text.ToString())) throw new ProviderRequestException("ResembleAI speech text cannot be empty.");
        payload!["data"] = JsonSerializer.SerializeToElement(text.ToString());
        return StreamSpeechCoreAsync(request.Model, payload, request.Instructions, request.IncludeRawChunks == true, headers, ct);
    }

    private async IAsyncEnumerable<StreamingSpeechPart> StreamSpeechCoreAsync(string model, Dictionary<string, JsonElement> payload,
        string? instructions, bool includeRaw, IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        using var request = new HttpRequestMessage(HttpMethod.Post, Synthesis + "stream") { Content = JsonContent.Create(payload, options: Json) };
        NativeHttp.ApplyHeaders(request, headers);
        Auth()(request.Headers);
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("audio/wav"));
        using var response = await NativeHttp.SendAsync(client, request, ct);
        var mime = response.Content.Headers.ContentType?.MediaType;
        if (mime is not (null or "audio/wav" or "audio/x-wav" or "audio/wave" or "application/octet-stream"))
            throw new IOException("ResembleAI did not return a WAV stream.");
        double? sampleRate = payload.TryGetValue("sample_rate", out var rate) && rate.ValueKind == JsonValueKind.Number && rate.TryGetDouble(out var hz) ? hz : null;
        var metadata = NativeMediaOptions.Metadata(Id, new { voice_uuid = OptionString(payload, "voice_uuid"), output_format = "wav", sample_rate = sampleRate });
        yield return new SpeechStreamStartPart { Warnings = string.IsNullOrWhiteSpace(instructions) ? [] : [new { type = "unsupported", feature = "instructions" }] };
        yield return new SpeechResponseMetadataPart { ModelId = model, Timestamp = DateTime.UtcNow, Headers = SafeHeaders(response) };
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        var buffer = new byte[16 * 1024];
        var received = false;
        while (await stream.ReadAsync(buffer, ct) is > 0 and var length)
        {
            received = true;
            var encoded = Convert.ToBase64String(buffer, 0, length);
            if (includeRaw) yield return new SpeechRawPart { RawValue = encoded };
            yield return new SpeechAudioPart { Audio = encoded, AudioFormat = new() { Type = "audio/wav", Rate = sampleRate }, ProviderMetadata = metadata };
        }
        if (!received) throw new IOException("ResembleAI returned an empty audio stream.");
        yield return new SpeechFinishPart();
    }

    private Dictionary<string, JsonElement> SpeechPayload(string model, string text, string? voice, string? format,
        Dictionary<string, JsonElement> options, double? rate, bool streaming)
    {
        if (!IsSpeechModel(model) || string.IsNullOrWhiteSpace(text))
            throw new ProviderRequestException("ResembleAI requires a Chatterbox speech model and nonempty text.");
        var parts = model.Split('/', 2);
        var modelVoice = parts.Length == 2 ? parts[1].Trim() : null;
        var nativeVoice = OptionString(options, "voice_uuid")?.Trim();
        voice = voice?.Trim();
        if (modelVoice is not null && (modelVoice.Length == 0 || modelVoice.Contains('/')
            || voice is { Length: > 0 } && !modelVoice.Equals(voice, StringComparison.OrdinalIgnoreCase)
            || nativeVoice is { Length: > 0 } && !modelVoice.Equals(nativeVoice, StringComparison.OrdinalIgnoreCase)))
            throw new ProviderRequestException("ResembleAI voice must match the voice UUID in the model.");
        var selected = modelVoice ?? voice ?? nativeVoice;
        if (string.IsNullOrWhiteSpace(selected)) throw new ProviderRequestException("ResembleAI requires a voice UUID.");
        var outputFormat = NormalizeFormat(format ?? OptionString(options, "output_format"));
        if (streaming && outputFormat is not (null or "wav")) throw new ProviderRequestException("ResembleAI streaming only supports WAV.");

        var payload = new Dictionary<string, JsonElement>(options, StringComparer.OrdinalIgnoreCase);
        payload.Remove("voice_uuid");
        payload.Remove("output_format");
        payload["voice_uuid"] = JsonSerializer.SerializeToElement(selected);
        payload["data"] = JsonSerializer.SerializeToElement(text);
        if (!streaming) payload["model"] = JsonSerializer.SerializeToElement(parts[0]);
        if (!streaming && outputFormat is not null) payload["output_format"] = JsonSerializer.SerializeToElement(outputFormat);
        if (rate is { } value)
        {
            payload.Remove("sample_rate");
            payload["sample_rate"] = JsonSerializer.SerializeToElement(value);
        }
        return payload;
    }

    private static Dictionary<string, JsonElement> SpeechOptions(Dictionary<string, JsonElement>? scoped)
        => NativeMediaOptions.Read(Id, scoped?.TryGetValue(Id, out var native) == true
            ? new Dictionary<string, JsonElement> { [Id] = native } : null, SpeechReserved);

    private static string? OptionString(IReadOnlyDictionary<string, JsonElement> options, string name)
    {
        var value = options.FirstOrDefault(x => x.Key.Equals(name, StringComparison.OrdinalIgnoreCase)).Value;
        return value.ValueKind switch { JsonValueKind.String => value.GetString(), JsonValueKind.Undefined or JsonValueKind.Null => null,
            _ => throw new ProviderRequestException($"ResembleAI {name} must be a string.") };
    }

    private static string? NormalizeFormat(string? format) => format?.Trim().ToLowerInvariant() switch
    {
        "wave" or "audio/wav" or "audio/x-wav" => "wav",
        "mpeg" or "audio/mpeg" => "mp3",
        "" or null => null,
        var other => other
    };
}
