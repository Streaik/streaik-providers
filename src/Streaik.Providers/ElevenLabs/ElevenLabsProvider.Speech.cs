using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.WebUtilities;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class ElevenLabsProvider
{
    private const int AudioLimit = 25 * 1024 * 1024;

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var plan = PlanSpeech(request, streaming: false);
        using var client = clients.CreateClient(Id);
        var upstream = await SendSpeechAsync(client, plan, headers, ct);
        if (!plan.Structured)
        {
            if (BinaryAudio(upstream.Content.Headers.ContentType?.MediaType)) return NormalizeAudio(upstream, plan.Format);
            upstream.Dispose();
            throw new IOException("ElevenLabs did not return audio.");
        }
        using (upstream)
        {
            var body = await NativeHttp.ReadAsync<JsonElement>(upstream, Json, ct);
            var data = String(body, "audio_base64") ?? String(body, "audio");
            if (!ValidAudio(data)) throw new IOException("ElevenLabs did not return valid audio.");
            var raw = JsonObject.Create(body)!;
            raw.Remove("audio_base64");
            raw.Remove("audio");
            var metadata = raw.Count == 0 ? null : Metadata(JsonSerializer.SerializeToElement(raw, Json));
            var result = new SpeechProviderResponse(new ByteArrayContent(Convert.FromBase64String(data!)), metadata);
            return NormalizeAudio(result, plan.Format);
        }
    }

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamSpeechCoreAsync(PlanSpeech(request, streaming: true), headers, ct);

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Text is not { Length: > 0 } || request.Text.Any(item => item?.Text is null))
            throw new ProviderRequestException("Speech requires text parts.");
        var fields = Options(ProviderOptions(request.ProviderOptions));
        var voices = request.Text.Select(item => item.Voice ?? request.Voice).Distinct().ToArray();
        if (IsDialogue(request.Model) && fields["inputs"] is null && (fields["dialogue"] as JsonObject)?["inputs"] is null)
        {
            var inputs = new JsonArray();
            foreach (var part in request.Text)
            {
                var item = Options(ProviderOptions(part.ProviderOptions));
                item["text"] = part.Text;
                item["voice_id"] = part.Voice ?? request.Voice;
                inputs.Add(item);
            }
            fields["inputs"] = inputs;
        }
        else if (request.Text.Any(part => part.ProviderOptions is not null) || (!IsDialogue(request.Model) && voices.Length != 1))
            throw new ProviderRequestException("ElevenLabs speech uses one voice unless a dialogue model is selected.");

        var format = request.OutputFormat?.Type;
        if (request.OutputFormat?.Rate is { } rate)
            format = format?.ToLowerInvariant() switch
            {
                "mp3" => $"mp3_{rate}_128", "opus" => $"opus_{rate}_128", "wav" => $"wav_{rate}",
                "pcm" or "pcm16" or "raw" => $"pcm_{rate}", "ulaw" or "mulaw" => $"ulaw_{rate}", "alaw" => $"alaw_{rate}",
                _ => throw new ProviderRequestException("Unsupported ElevenLabs audio format with a custom rate.")
            };
        return StreamSpeechAsync(new AudioSpeechRequest { Model = request.Model, Input = string.Concat(request.Text.Select(item => item.Text)),
            Voice = voices.FirstOrDefault(), ResponseFormat = format, Speed = request.Speed,
            AdditionalProperties = fields.Deserialize<Dictionary<string, JsonElement>>(Json) }, headers, ct);
    }

    private async IAsyncEnumerable<StreamingSpeechPart> StreamSpeechCoreAsync(SpeechPlan plan, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        using var upstream = await SendSpeechAsync(client, plan, headers, ct);
        var contentType = upstream.Content.Headers.ContentType;
        if (!plan.Structured && !BinaryAudio(contentType?.MediaType)) throw new IOException("ElevenLabs did not return audio.");
        var mime = SpeechMimeType(contentType?.MediaType, plan.Format);
        var codec = plan.Format.Split('_')[0];
        var rate = plan.Format.Split('_').Skip(1).FirstOrDefault();
        var format = new AudioFormat { Type = codec == "auto" ? (mime == "audio/mpeg" ? "mp3" : mime.Split('/')[1]) : codec,
            Rate = int.TryParse(rate, out var samples) ? samples : null };
        await using var stream = await upstream.Content.ReadAsStreamAsync(ct);
        yield return new SpeechStreamStartPart();
        yield return new SpeechResponseMetadataPart { Timestamp = DateTime.UtcNow };

        if (plan.Structured && contentType?.MediaType?.StartsWith("multipart/", StringComparison.OrdinalIgnoreCase) == true)
        {
            var boundary = contentType.Parameters.FirstOrDefault(p => p.Name.Equals("boundary", StringComparison.OrdinalIgnoreCase))?.Value?.Trim('"');
            if (string.IsNullOrWhiteSpace(boundary) || boundary.Length > 128) throw new IOException("Invalid ElevenLabs multipart audio boundary.");
            var reader = new MultipartReader(boundary, stream) { BodyLengthLimit = AudioLimit };
            while (await reader.ReadNextSectionAsync(ct) is { } section)
            {
                if (section.ContentType?.StartsWith("application/json", StringComparison.OrdinalIgnoreCase) == true)
                {
                    using var document = await JsonDocument.ParseAsync(section.Body, cancellationToken: ct);
                    foreach (var part in StructuredParts(document.RootElement, format)) yield return part;
                }
                else if (section.ContentType?.StartsWith("audio/", StringComparison.OrdinalIgnoreCase) == true || section.ContentType == "application/octet-stream")
                    await foreach (var part in AudioParts(section.Body, format, ct)) yield return part;
            }
        }
        else if (plan.Structured && !(contentType?.MediaType?.StartsWith("audio/", StringComparison.OrdinalIgnoreCase) == true))
        {
            using var reader = new StreamReader(stream);
            while (await reader.ReadLineAsync(ct) is { } line)
            {
                var data = line.Trim();
                if (data.StartsWith("data:", StringComparison.Ordinal)) data = data[5..].TrimStart();
                else if (data.StartsWith(':') || data.StartsWith("event:", StringComparison.Ordinal) || data.StartsWith("id:", StringComparison.Ordinal)
                    || data.StartsWith("retry:", StringComparison.Ordinal)) continue;
                if (data == "[DONE]") break;
                if (data.Length == 0) continue;
                if (ValidAudio(data)) { yield return new SpeechAudioPart { Audio = data, AudioFormat = format }; continue; }
                using var document = JsonDocument.Parse(data);
                foreach (var part in StructuredParts(document.RootElement, format)) yield return part;
            }
        }
        else
        {
            await foreach (var part in AudioParts(stream, format, ct)) yield return part;
        }
        ct.ThrowIfCancellationRequested();
        yield return new SpeechFinishPart();
    }

    private static IEnumerable<StreamingSpeechPart> StructuredParts(JsonElement root, AudioFormat format)
    {
        var metadata = Copy(root);
        var audio = new List<string>();
        ReadAudio(metadata);
        var raw = JsonSerializer.SerializeToElement(metadata, Json);
        var hasMetadata = metadata is not JsonObject { Count: 0 };
        foreach (var data in audio) yield return new SpeechAudioPart { Audio = data, AudioFormat = format, ProviderMetadata = hasMetadata ? Metadata(raw) : null };
        if (audio.Count == 0 && hasMetadata) yield return new SpeechRawPart { RawValue = raw };

        void ReadAudio(JsonNode? value)
        {
            if (value is JsonObject obj)
                foreach (var pair in obj.ToArray())
                {
                    if (pair.Key is "audio_base64" or "audio" && Text(pair.Value) is { } bytes)
                    {
                        if (!ValidAudio(bytes)) throw new IOException("ElevenLabs returned invalid audio in a stream event.");
                        audio.Add(bytes);
                        obj.Remove(pair.Key);
                    }
                    else ReadAudio(pair.Value);
                }
            else if (value is JsonArray array)
                foreach (var item in array) ReadAudio(item);
        }
    }

    private static async IAsyncEnumerable<SpeechAudioPart> AudioParts(Stream stream, AudioFormat format, [EnumeratorCancellation] CancellationToken ct)
    {
        var buffer = new byte[16 * 1024];
        while (await stream.ReadAsync(buffer, ct) is > 0 and var length)
            yield return new SpeechAudioPart { Audio = Convert.ToBase64String(buffer, 0, length), AudioFormat = format };
    }

    private Task<HttpResponseMessage> SendSpeechAsync(HttpClient client, SpeechPlan plan, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SendAsync(client, HttpMethod.Post, plan.Path, JsonContent.Create(plan.Body, options: Json), headers, ct,
            plan.Structured ? "application/json, text/event-stream, multipart/mixed, audio/*" : "audio/*");

    private static SpeechPlan PlanSpeech(AudioSpeechRequest request, bool streaming)
    {
        if (!IsRequestModel(request.Model)) throw new ProviderRequestException("Invalid ElevenLabs speech model.");
        var body = Options(request.AdditionalProperties);
        var model = request.Model;
        var dialogue = IsDialogue(model);
        var music = IsMusic(model);
        var sound = model == SoundModel;
        var format = OutputFormat(request.ResponseFormat ?? Text(body["output_format"]), music);
        var query = new List<string> { $"output_format={Uri.EscapeDataString(format)}" };
        if (TakeQuery(body, "enable_logging") is { } logging) query.Add(logging);
        if (TakeQuery(body, "optimize_streaming_latency") is { } latency && !music && !dialogue && !sound) query.Add(latency);
        var voice = request.Voice ?? Text(body["voice_id"]) ?? Text(body["voice"]);
        var language = Text(body["language"]) ?? Text(body["language_code"]);
        var timestamps = Flag(body["with_timestamps"]);
        var waveform = Flag(body["with_waveform_visual"]);
        var detailed = Flag(body["detailed_stream"]);
        var dialogueOptions = body["dialogue"] as JsonObject;
        var soundOptions = body["text_to_sound"] as JsonObject;
        foreach (var key in new[] { "model", "model_id", "text", "prompt", "input", "voice", "voice_id", "speed", "instructions", "output_format",
            "response_format", "stream_format", "stream", "language", "language_code", "with_timestamps", "with_waveform_visual", "detailed_stream", "dialogue", "text_to_sound" })
            body.Remove(key);
        body["model_id"] = dialogue ? model[..^DialogueSuffix.Length] : model;
        if (!music && !sound && !string.IsNullOrWhiteSpace(language)) body["language_code"] = language;

        string path;
        var structured = false;
        if (dialogue)
        {
            body["inputs"] ??= dialogueOptions?["inputs"]?.DeepClone();
            body["settings"] ??= dialogueOptions?["settings"]?.DeepClone();
            if (body["inputs"] is not JsonArray { Count: > 0 } inputs || inputs.Any(input => input is not JsonObject item
                || string.IsNullOrWhiteSpace(Text(item["text"])) || string.IsNullOrWhiteSpace(Text(item["voice_id"]))))
                throw new ProviderRequestException("ElevenLabs dialogue requires inputs with text and voice_id.");
            if (inputs.Select(input => Text(input!["voice_id"])).Distinct().Count() > 10 || inputs.Sum(input => (long)Text(input!["text"])!.Length) > 2000)
                throw new ProviderRequestException("ElevenLabs dialogue supports at most 10 voices and 2,000 characters.");
            path = "text-to-dialogue" + (streaming ? "/stream" : "") + (timestamps ? "/with-timestamps" : "");
            structured = timestamps;
        }
        else if (music)
        {
            if (body["composition_plan"] is null) body["prompt"] = RequireText(request.Input);
            structured = streaming && (detailed || timestamps || waveform);
            if (structured)
            {
                if (timestamps) body["with_timestamps"] = true;
                if (waveform) body["with_waveform_visual"] = true;
            }
            path = structured ? "music/detailed/stream" : streaming ? "music/stream" : "music";
        }
        else if (sound)
        {
            body["text"] = RequireText(request.Input);
            if (soundOptions is not null)
                foreach (var key in new[] { "loop", "duration_seconds", "prompt_influence" }) body[key] ??= soundOptions[key]?.DeepClone();
            path = "sound-generation"; // This endpoint returns binary audio; the gateway supplies client framing.
        }
        else
        {
            if (string.IsNullOrWhiteSpace(voice) || voice.Length > 128 || !voice.All(ch => char.IsAsciiLetterOrDigit(ch) || ch is '-' or '_'))
                throw new ProviderRequestException("An ElevenLabs voice ID is required for text-to-speech.");
            body["text"] = RequireText(request.Input);
            if (request.Speed is { } speed)
            {
                if (!double.IsFinite(speed) || speed <= 0) throw new ProviderRequestException("ElevenLabs speech speed must be positive.");
                var settings = body["voice_settings"] as JsonObject ?? new JsonObject();
                foreach (var key in settings.Select(entry => entry.Key).Where(key => key.Equals("speed", StringComparison.OrdinalIgnoreCase)).ToArray())
                    settings.Remove(key);
                settings["speed"] = speed;
                if (settings.Parent is null) body["voice_settings"] = settings;
            }
            path = $"text-to-speech/{voice}" + (streaming ? "/stream" : "") + (timestamps ? "/with-timestamps" : "");
            structured = timestamps;
        }
        return new(path + "?" + string.Join('&', query), body, format, structured);
    }

    private static bool IsRequestModel(string? model) => model is { Length: > 0 and <= 128 };
    private static string RequireText(string? value) => !string.IsNullOrWhiteSpace(value) ? value : throw new ProviderRequestException("ElevenLabs speech requires text.");
    private static IReadOnlyDictionary<string, JsonElement>? ProviderOptions(IReadOnlyDictionary<string, JsonElement>? source)
        => source?.GetValueOrDefault(Id) switch
        {
            null or { ValueKind: JsonValueKind.Null } => null,
            { ValueKind: JsonValueKind.Object } value => value.Deserialize<Dictionary<string, JsonElement>>(Json),
            _ => throw new ProviderRequestException("ElevenLabs provider options must be an object.")
        };
    private static bool ValidAudio(string? value) => value is { Length: > 0 } && value.Length <= ((long)AudioLimit + 2) / 3 * 4
        && System.Buffers.Text.Base64.IsValid(value, out var length) && length is > 0 and <= AudioLimit;
    private static string OutputFormat(string? format, bool music) => format?.Trim().ToLowerInvariant() switch
    {
        null or "" => music ? "auto" : "mp3_44100_128", "mp3" => "mp3_44100_128", "opus" => "opus_48000_128", "wav" => "wav_44100",
        "pcm" or "pcm16" or "raw" => "pcm_44100", "ulaw" or "mulaw" => "ulaw_8000", "alaw" => "alaw_8000", var value => value
    };
    private static string SpeechMimeType(string? type, string format) => type is null or "application/json" or "application/octet-stream" or "text/event-stream"
        || type.StartsWith("multipart/", StringComparison.OrdinalIgnoreCase) ? format.Split('_')[0] switch
        { "mp3" => "audio/mpeg", "opus" => "audio/ogg", "wav" => "audio/wav", "ulaw" or "alaw" => "audio/basic", _ => "application/octet-stream" } : type;
    private static bool BinaryAudio(string? type) => type is null or "application/octet-stream" || type.StartsWith("audio/", StringComparison.OrdinalIgnoreCase);
    private static HttpResponseMessage NormalizeAudio(HttpResponseMessage response, string format)
    {
        response.Content.Headers.ContentType = new(SpeechMimeType(response.Content.Headers.ContentType?.MediaType, format));
        return response;
    }

    private sealed record SpeechPlan(string Path, JsonObject Body, string Format, bool Structured);
}
