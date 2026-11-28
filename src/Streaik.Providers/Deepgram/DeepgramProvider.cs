using System.Globalization;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Deepgram;

public static class DeepgramRegistration
{
    public static IServiceCollection AddDeepgram(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, DeepgramProvider.Id, "https://api.deepgram.com/");
        return services.AddProvider<DeepgramProvider>(DeepgramProvider.Id);
    }
}

public sealed class DeepgramProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : ISpeechProvider, ITranscriptionProvider, IStreamingTranscriptionProvider, IRealtimeTokenProvider, IModelListingProvider
{
    public const string Id = "deepgram";
    private const string RealtimeSpeechModel = "aura-2-asteria-en";
    private static readonly string[] TranscriptionOptions = ["language", "diarize", "detect_language", "smart_format", "punctuate", "utterances", "numerals", "profanity_filter", "encoding", "sample_rate", "channels", "keyterm", "keywords", "mip_opt_out"];

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        foreach (var (id, type) in new[] { (RealtimeSpeechModel, "speech"), ("aura-2-thalia-en", "speech"), ("nova-3", "transcription"), ("nova-2", "transcription") })
        {
            ct.ThrowIfCancellationRequested();
            yield return new Model { Id = id, Name = id, Type = type, Tags = id == RealtimeSpeechModel ? ["realtime", "realtime-token"]
                : type == "transcription" ? ["realtime"] : null };
        }
    }

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.StreamFormat is not (null or "audio") || request.Instructions is { Length: > 0 })
            throw new ProviderRequestException("Deepgram speech does not accept instructions or OpenAI SSE. Use /api/speech/stream for streamed audio.", 422);
        var model = request.Voice is { Length: > 0 } voice && voice.StartsWith("aura-", StringComparison.Ordinal) ? voice : request.Model;
        if (request.Voice is { Length: > 0 } && model == request.Model && request.Voice != model)
            throw new ProviderRequestException("Use a Deepgram Aura model ID as the voice.", 422);
        var query = Options(request.AdditionalProperties, ["sample_rate", "bit_rate", "mip_opt_out"]);
        query["model"] = model;
        var format = request.ResponseFormat ?? "mp3";
        (query["encoding"], var container, var mediaType) = format switch
        {
            "mp3" => ("mp3", null, "audio/mpeg"), "opus" => ("opus", "ogg", "audio/ogg"),
            "wav" => ("linear16", "wav", "audio/wav"), "pcm" or "pcm16" => ("linear16", "none", "application/octet-stream"),
            "flac" => ("flac", null, "audio/flac"), _ => throw new ProviderRequestException("Unsupported Deepgram audio format.", 422)
        };
        if (container is not null) query["container"] = container;
        if (request.Speed is { } speed) query["speed"] = speed.ToString(CultureInfo.InvariantCulture);
        var key = RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        var response = await SendAsync(client, HttpMethod.Post, "v1/speak?" + Query(query), JsonContent.Create(new { text = request.Input }), "Token", key, headers, ct);
        response.Content.Headers.ContentType ??= new(mediaType);
        return response;
    }

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName, string? language,
        IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (options?.Keys.Any(x => x is "prompt" or "temperature" or "timestamp_granularities[]") == true)
            throw new ProviderRequestException("Deepgram does not support OpenAI prompt, temperature or timestamp granularity options.", 422);
        var query = Options(options, TranscriptionOptions);
        query["model"] = model;
        if (!string.IsNullOrWhiteSpace(language)) query["language"] = language;
        var key = RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        using var body = new StreamContent(audio);
        body.Headers.ContentType = MediaTypeHeaderValue.TryParse(mediaType, out var type) ? type : new("application/octet-stream");
        var root = await JsonAsync<JsonElement>(client, HttpMethod.Post, "v1/listen?" + Query(query), body, "Token", key, headers, ct);
        var channels = Array(Property(Property(root, "results"), "channels")).ToArray();
        if (channels.Length == 0) throw new JsonException("Deepgram returned no transcript channel.");
        var alternatives = channels.Select(c => Array(Property(c, "alternatives")).FirstOrDefault()).ToArray();
        var segments = alternatives.SelectMany(a => Array(Property(a, "words"))).Select(w => new TranscriptionSegment
        { Text = String(w, "punctuated_word") ?? String(w, "word") ?? "", StartSecond = Number(w, "start") ?? 0, EndSecond = Number(w, "end") ?? 0 });
        return new() { Text = string.Join(' ', alternatives.Select(a => String(a, "transcript"))), Language = String(channels[0], "detected_language") ?? language,
            DurationInSeconds = Number(Property(root, "metadata"), "duration"), Segments = segments.ToArray() };
    }

    public async IAsyncEnumerable<StreamingTranscriptionPart> StreamTranscriptionAsync(StreamingTranscriptionRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        byte[] audio;
        try { audio = Convert.FromBase64String(request.Audio); }
        catch (FormatException) { throw new ProviderRequestException("Invalid base64 audio."); }
        if (audio.Length is 0 or > 25 * 1024 * 1024) throw new ProviderRequestException("Invalid audio length.", 413);
        var query = Options(ProviderOptions(request.ProviderOptions), TranscriptionOptions);
        query["model"] = request.Model;
        query["interim_results"] = "true";
        if (request.InputAudioFormat.Type is "pcm" or "pcm16" or "linear16" or "mulaw" or "alaw")
        {
            query["encoding"] = request.InputAudioFormat.Type is "pcm" or "pcm16" ? "linear16" : request.InputAudioFormat.Type;
            if (request.InputAudioFormat.Rate is not (>= 8000 and <= 96000)) throw new ProviderRequestException("Raw audio requires a valid sample rate.");
            query["sample_rate"] = request.InputAudioFormat.Rate.Value.ToString(CultureInfo.InvariantCulture);
        }
        var key = RequiredKey(credentials, Id);
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromMinutes(2));
        var socketCt = timeout.Token;
        using var client = clients.CreateClient(Id);
        using var socket = new ClientWebSocket();
        socket.Options.SetRequestHeader("Authorization", "Token " + key);
        var url = new UriBuilder(client.BaseAddress!) { Scheme = "wss", Path = "/v1/listen", Query = Query(query) };
        await socket.ConnectAsync(url.Uri, client, socketCt);
        var send = SendAudioAsync(socket, audio, timeout);
        try
        {
            yield return new TranscriptionStreamStartPart();
            yield return new TranscriptionResponseMetadataPart { Timestamp = DateTime.UtcNow, ModelId = request.Model };
            var segments = new List<TranscriptionSegment>();
            double? duration = null;
            var gotMetadata = false;
            while (await ReadMessageAsync(socket, socketCt) is { } root)
            {
                if (request.IncludeRawChunks == true) yield return new TranscriptionRawPart { RawValue = root };
                switch (String(root, "type"))
                {
                    case "Metadata": gotMetadata = true; duration = Number(root, "duration"); break;
                    case "Error": throw new IOException("Deepgram returned a transcription stream error.");
                    case "Results":
                        var first = Array(Property(Property(root, "channel"), "alternatives")).FirstOrDefault();
                        if (String(first, "transcript") is not { Length: > 0 } text) break;
                        var start = Number(root, "start") ?? 0;
                        var length = Number(root, "duration") ?? 0;
                        if (Property(root, "is_final").ValueKind == JsonValueKind.True)
                        {
                            segments.Add(new() { Text = text, StartSecond = start, EndSecond = start + length });
                            yield return new TranscriptionFinalPart { Text = text, StartSecond = start, EndSecond = start + length };
                        }
                        else yield return new TranscriptionPartialPart { Text = text, StartSecond = start, DurationInSeconds = length };
                        break;
                }
            }
            await send;
            if (!gotMetadata) throw new IOException("Deepgram did not send terminal transcription metadata.");
            yield return new TranscriptionFinishPart { Text = string.Join(' ', segments.Select(s => s.Text)), Segments = segments, DurationInSeconds = duration };
        }
        finally
        {
            await timeout.CancelAsync();
            try { await send; } catch (Exception e) when (e is WebSocketException or OperationCanceledException or IOException) { }
        }
    }

    public async Task<RealtimeTokenResponse> CreateRealtimeTokenAsync(string model, RealtimeTokenRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var settings = ProviderOptions(request.SessionConfig?.ProviderOptions);
        var ttl = 30;
        if (settings?.TryGetValue("ttl_seconds", out var value) == true && (value.ValueKind != JsonValueKind.Number || !value.TryGetInt32(out ttl) || ttl is < 1 or > 3600))
            throw new ProviderRequestException("Deepgram ttl_seconds must be an integer from 1 to 3600.");
        var key = RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        var response = await JsonAsync<JsonElement>(client, HttpMethod.Post, "v1/auth/grant", JsonContent.Create(new { ttl_seconds = ttl }), "Token", key, headers, ct);
        var path = model == RealtimeSpeechModel ? "/v1/speak" : "/v1/listen";
        var url = new UriBuilder(client.BaseAddress!) { Scheme = "wss", Path = path, Query = "model=" + Uri.EscapeDataString(model) };
        return new() { Token = RealtimeOptions.Required(response, "access_token", key), Url = url.Uri.AbsoluteUri,
            ExpiresAt = DateTimeOffset.UtcNow.AddSeconds(Number(response, "expires_in") ?? ttl).ToUnixTimeSeconds() };
    }

    private static async Task SendAudioAsync(ClientWebSocket socket, byte[] audio, CancellationTokenSource timeout)
    {
        try
        {
            for (var offset = 0; offset < audio.Length; offset += 8192)
                await socket.SendAsync(audio.AsMemory(offset, Math.Min(audio.Length - offset, 8192)), WebSocketMessageType.Binary, true, timeout.Token);
            await socket.SendAsync("{\"type\":\"CloseStream\"}"u8.ToArray().AsMemory(), WebSocketMessageType.Text, true, timeout.Token);
        }
        catch { await timeout.CancelAsync(); throw; }
    }

    private static async Task<JsonElement?> ReadMessageAsync(ClientWebSocket socket, CancellationToken ct)
    {
        using var buffer = new MemoryStream();
        var bytes = new byte[4096];
        while (true)
        {
            var result = await socket.ReceiveAsync(bytes.AsMemory(), ct);
            if (result.MessageType == WebSocketMessageType.Close) return null;
            if (result.MessageType != WebSocketMessageType.Text || buffer.Length + result.Count > 1024 * 1024)
                throw new IOException("Unexpected Deepgram message.");
            buffer.Write(bytes, 0, result.Count);
            if (result.EndOfMessage) return JsonSerializer.Deserialize<JsonElement>(buffer.GetBuffer().AsSpan(0, (int)buffer.Length));
        }
    }

    private static double? Number(JsonElement root, string name) => Property(root, name) is { ValueKind: JsonValueKind.Number } n ? n.GetDouble() : null;
    private static IReadOnlyDictionary<string, JsonElement>? ProviderOptions(Dictionary<string, JsonElement>? options)
        => options?.TryGetValue(Id, out var selected) == true && selected.ValueKind == JsonValueKind.Object
            ? selected.EnumerateObject().ToDictionary(x => x.Name, x => x.Value) : null;

    private static Dictionary<string, string> Options(IReadOnlyDictionary<string, JsonElement>? values, string[] allowed)
        => values?.Where(x => allowed.Contains(x.Key, StringComparer.Ordinal) && x.Value.ValueKind is JsonValueKind.String or JsonValueKind.Number or JsonValueKind.True or JsonValueKind.False)
            .ToDictionary(x => x.Key, x => x.Value.ValueKind == JsonValueKind.String ? x.Value.GetString()! : x.Value.GetRawText()) ?? new();

    private static string Query(Dictionary<string, string> values) => string.Join('&', values.Select(x => $"{x.Key}={Uri.EscapeDataString(x.Value)}"));
}
