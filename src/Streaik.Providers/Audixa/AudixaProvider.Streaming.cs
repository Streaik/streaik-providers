using System.Buffers.Binary;
using System.Net.WebSockets;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class AudixaProvider
{
    private const int SampleRate = 24_000;

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct) => StreamAsync(request, headers, false, ct);

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Text is not { Length: > 0 } || request.Text.Any(part => part?.Text is null || part.ProviderOptions is not null)
            || request.Text.Select(part => part.Voice ?? request.Voice).Distinct().Count() != 1)
            throw new ProviderRequestException("Audixa speech streams support one voice without per-part options.");
        if (request.OutputFormat?.Rate is { } rate && rate != SampleRate)
            throw new ProviderRequestException("Audixa speech streams use a fixed sample rate of 24000 Hz.");

        var options = new Dictionary<string, JsonElement>(StringComparer.OrdinalIgnoreCase);
        if (request.ProviderOptions?.TryGetValue(Id, out var native) == true)
        {
            if (native.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Audixa options must be an object.");
            foreach (var property in native.EnumerateObject()) options[property.Name] = property.Value;
        }
        if (request.Language is { } language) options["language"] = JsonSerializer.SerializeToElement(language);
        return StreamAsync(new AudioSpeechRequest { Model = request.Model, Input = string.Concat(request.Text.Select(part => part.Text)),
            Voice = request.Text[0].Voice ?? request.Voice, ResponseFormat = request.OutputFormat?.Type ?? TakeString(options, "audio_format") ?? "raw",
            Instructions = request.Instructions, Speed = request.Speed, AdditionalProperties = options }, headers, request.IncludeRawChunks == true, ct);
    }

    private async IAsyncEnumerable<StreamingSpeechPart> StreamAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers,
        bool includeRaw, [EnumeratorCancellation] CancellationToken ct)
    {
        var payload = Payload(request);
        var format = TakeString(payload, "audio_format")?.Trim().ToLowerInvariant();
        var warnings = string.IsNullOrWhiteSpace(request.Instructions) ? System.Array.Empty<object>() : [new { type = "unsupported", feature = "instructions" }];
        if (format is not ("pcm" or "raw" or "pcm16"))
        {
            // Audixa's socket always emits raw float PCM; use completed HTTP audio for file formats.
            using var response = await SpeakAsync(request, headers, ct);
            var audio = await response.Content.ReadAsByteArrayAsync(ct);
            yield return new SpeechStreamStartPart { Warnings = warnings };
            yield return new SpeechResponseMetadataPart { ModelId = request.Model, Timestamp = DateTime.UtcNow };
            if (audio.Length != 0)
            {
                var encoded = Convert.ToBase64String(audio);
                if (includeRaw) yield return new SpeechRawPart { RawValue = encoded };
                yield return new SpeechAudioPart { Audio = encoded, AudioFormat = new() { Type = format ?? response.Content.Headers.ContentType?.MediaType ?? "audio/wav" } };
            }
            yield return new SpeechFinishPart { ProviderMetadata = (response as SpeechProviderResponse)?.ProviderMetadata };
            yield break;
        }

        payload["audio_format"] = JsonSerializer.SerializeToElement("wav");
        using var client = clients.CreateClient(Id);
        using var socket = new ClientWebSocket();
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromMinutes(2));
        var socketCt = timeout.Token;
        var endpoint = new UriBuilder(client.BaseAddress!) { Scheme = "wss", Path = "/v3/tts/stream",
            Query = "api_key=" + Uri.EscapeDataString(RequiredKey(credentials, Id)) };
        try { await socket.ConnectAsync(endpoint.Uri, client, socketCt); }
        catch (WebSocketException) { throw new HttpRequestException("Could not open the Audixa speech stream."); }
        await socket.SendAsync(JsonSerializer.SerializeToUtf8Bytes(payload, Json).AsMemory(), WebSocketMessageType.Text, true, socketCt);

        yield return new SpeechStreamStartPart { Warnings = warnings };
        yield return new SpeechResponseMetadataPart { ModelId = request.Model, Timestamp = DateTime.UtcNow };
        var buffer = new byte[16 * 1024];
        while (true)
        {
            var (type, bytes) = await ReceiveAsync(socket, buffer, socketCt);
            if (type == WebSocketMessageType.Binary)
            {
                if (bytes.Length == 0) continue;
                if (includeRaw) yield return new SpeechRawPart { RawValue = Convert.ToBase64String(bytes) };
                yield return new SpeechAudioPart { Audio = Convert.ToBase64String(format == "pcm16" ? ToPcm16(bytes) : bytes),
                    AudioFormat = new() { Type = format, Rate = SampleRate } };
                continue;
            }
            using var control = JsonDocument.Parse(bytes);
            var message = control.RootElement;
            if (includeRaw) yield return new SpeechRawPart { RawValue = message.Clone() };
            switch (String(message, "type"))
            {
                case "started": break;
                case "completed":
                    if (socket.State == WebSocketState.Open)
                        await socket.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, null, socketCt);
                    yield return new SpeechFinishPart { ProviderMetadata = Metadata(message) };
                    yield break;
                case "error": throw new IOException("Audixa TTS stream failed: " + (String(message, "message") ?? String(message, "error") ?? "unknown error"));
                default: throw new JsonException("Audixa returned an unknown speech stream event.");
            }
        }
    }

    private static async Task<(WebSocketMessageType Type, byte[] Bytes)> ReceiveAsync(WebSocket socket, byte[] buffer, CancellationToken ct)
    {
        using var message = new MemoryStream();
        WebSocketMessageType? type = null;
        while (true)
        {
            var frame = await socket.ReceiveAsync(buffer.AsMemory(), ct);
            if (frame.MessageType == WebSocketMessageType.Close) throw new IOException("Audixa closed the speech stream before completion.");
            if ((type is not null && type != frame.MessageType) || message.Length + frame.Count > 4 * 1024 * 1024)
                throw new IOException("Audixa sent an invalid speech stream message.");
            type = frame.MessageType;
            message.Write(buffer, 0, frame.Count);
            if (frame.EndOfMessage) return (type.Value, message.ToArray());
        }
    }

    private static byte[] ToPcm16(byte[] bytes)
    {
        if (bytes.Length % sizeof(float) != 0) throw new IOException("Audixa returned an incomplete PCM sample.");
        var pcm = new byte[bytes.Length / 2];
        for (var offset = 0; offset < bytes.Length; offset += sizeof(float))
        {
            var sample = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(offset));
            var value = float.IsNaN(sample) ? 0f : Math.Clamp(sample * 32768f, short.MinValue, short.MaxValue);
            BinaryPrimitives.WriteInt16LittleEndian(pcm.AsSpan(offset / 2), (short)value);
        }
        return pcm;
    }
}
