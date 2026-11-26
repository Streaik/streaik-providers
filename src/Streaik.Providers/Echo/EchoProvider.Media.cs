using System.Buffers.Binary;
using System.Net;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Streaik.OpenAI.Model.Providers.OpenAI.Live;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;

namespace Streaik.Providers;

public sealed partial class EchoProvider : IImageModelRoutingProvider, IVideoProvider, ISpeechProvider, IStreamingSpeechEventsProvider,
    ITranscriptionProvider, IStreamingTranscriptionProvider, INativeAudioTranscriptionProvider, IEmbeddingsProvider, IRerankingProvider,
    IRealtimeTokenProvider, IRealtimeSessionProvider, IOpenAIRealtimeProvider, IOpenAILiveSessionProvider
{
    private const string ImageModel = "echo-image", VideoModel = "echo-video", SpeechModel = "echo-speech", TranscriptionModel = "echo-transcription",
        EmbeddingModel = "echo-embedding", RerankModel = "echo-rerank", RealtimeModel = "echo-realtime";
    private const string Png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNQdUj+DwADHgHItgWuOwAAAABJRU5ErkJggg==";
    private const string VideoOperation = "echo-fixture-video";
    private const string Transcript = "Echo transcription fixture.";
    private const string FixtureToken = "echo_fixture_not_connectable";
    private const string FixtureUrl = "wss://echo-fixture.invalid/not-connectable";
    private const string RejectedSdp = "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=Echo fixture (not connectable)\r\nt=0 0\r\n";
    private static readonly Lazy<byte[]> Video = new(() =>
    {
        using var resource = typeof(EchoProvider).Assembly.GetManifestResourceStream("Streaik.Providers.Echo.EchoFixture.webm")!;
        using var output = new MemoryStream();
        resource.CopyTo(output);
        return output.ToArray();
    });

    private static void CheckModel(string? model, string expected, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        if (model != expected) throw new ProviderRequestException($"Use echo/{expected} for this endpoint.", 404);
    }

    public Task<bool> IsImageModelAsync(string model, CancellationToken ct) => Task.FromResult(model == ImageModel);

    private static OpenAIImagesResponse Image(string model, string? format, int? count, bool? stream, CancellationToken ct)
    {
        CheckModel(model, ImageModel, ct);
        if (format is not (null or "png")) throw new ProviderRequestException("Echo images support PNG only.");
        var n = count ?? 1;
        if (n is < 1 or > 10 || stream == true && n != 1) throw new ProviderRequestException("Invalid Echo image count.");
        return new() { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), OutputFormat = "png",
            Data = Enumerable.Range(0, n).Select(_ => new OpenAIImageData { B64Json = Png }).ToList() };
    }

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => Task.FromResult(Image(request.Model, request.OutputFormat, request.N, request.Stream, ct));

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => Task.FromResult(Image(request.Model, request.OutputFormat, request.N, request.Stream, ct));

    public Task<VideoOperationStartResult> StartVideoAsync(VideoRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        CheckModel(request.Model, VideoModel, ct);
        return Task.FromResult(new VideoOperationStartResult { Operation = VideoOperation, Response = new() });
    }

    public Task<VideoOperationStatusResult> GetVideoAsync(string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        if (operation != VideoOperation) throw new ProviderRequestException("Unknown Echo video operation.", 404);
        return Task.FromResult<VideoOperationStatusResult>(new VideoOperationCompletedResult { Response = new(),
            Videos = [new() { Type = "base64", MediaType = "video/webm", Data = Convert.ToBase64String(Video.Value) }] });
    }

    // A small PCM WAV tone, not a speech synthesizer or a copy of the input audio.
    private static byte[] Tone(bool wav)
    {
        const int sampleRate = 24000, samples = 2400;
        var bytes = new byte[2 * samples + (wav ? 44 : 0)];
        if (wav)
        {
            "RIFF"u8.CopyTo(bytes);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(4), bytes.Length - 8);
            "WAVEfmt "u8.CopyTo(bytes.AsSpan(8));
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(16), 16);
            BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(20), 1);
            BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(22), 1);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(24), sampleRate);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(28), sampleRate * 2);
            BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(32), 2);
            BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(34), 16);
            "data"u8.CopyTo(bytes.AsSpan(36));
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(40), samples * 2);
        }
        for (var i = 0; i < samples; i++)
            BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan((wav ? 44 : 0) + 2 * i), (short)(2000 * Math.Sin(2 * Math.PI * 440 * i / sampleRate)));
        return bytes;
    }

    private static bool IsWav(string? format) => format switch
    {
        null or "wav" => true,
        "pcm" => false,
        _ => throw new ProviderRequestException("Echo speech supports wav and pcm only.")
    };

    public Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        CheckModel(request.Model, SpeechModel, ct);
        var wav = IsWav(request.ResponseFormat);
        var audio = new ByteArrayContent(Tone(wav));
        audio.Headers.ContentType = new MediaTypeHeaderValue(wav ? "audio/wav" : "application/octet-stream");
        return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = audio });
    }

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeechParts(request.Model, request.ResponseFormat, request.Input?.Length ?? 0, ct);

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeechParts(request.Model, request.OutputFormat?.Type, request.Text.Sum(part => part.Text?.Length ?? 0), ct);

    private static async IAsyncEnumerable<StreamingSpeechPart> SpeechParts(string model, string? format, int characters, [EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        CheckModel(model, SpeechModel, ct);
        var wav = IsWav(format);
        yield return new SpeechStreamStartPart();
        yield return new SpeechResponseMetadataPart { ModelId = "echo/" + model };
        yield return new SpeechAudioPart { Audio = Convert.ToBase64String(Tone(wav)), AudioFormat = new() { Type = wav ? "wav" : "pcm" } };
        yield return new SpeechFinishPart { Usage = new() { InputCharacters = characters } };
    }

    private static TranscriptionSegment Segment() => new() { Text = Transcript, StartSecond = 0, EndSecond = 0.1 };

    public Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName, string? language,
        IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        CheckModel(model, TranscriptionModel, ct);
        return Task.FromResult(new TranscriptionResponse { Text = Transcript, Language = "en", DurationInSeconds = 0.1, Segments = [Segment()], Response = new() });
    }

    public async IAsyncEnumerable<StreamingTranscriptionPart> StreamTranscriptionAsync(StreamingTranscriptionRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        CheckModel(request.Model, TranscriptionModel, ct);
        yield return new TranscriptionResponseMetadataPart { ModelId = request.Model };
        yield return new TranscriptionDeltaPart { Delta = Transcript };
        yield return new TranscriptionFinishPart { Text = Transcript, Language = "en", DurationInSeconds = 0.1, Segments = [Segment()] };
    }

    public Task<HttpResponseMessage> TranscribeNativeAsync(string model, IFormCollection request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        CheckModel(model, TranscriptionModel, ct);
        var format = request["response_format"].ToString();
        var stream = request["stream"] == "true";
        if (stream && format is not ("" or "json")) throw new ProviderRequestException("Streaming transcription requires JSON response format.");
        var (body, type) = (stream, format) switch
        {
            (true, _) => ($"data: {JsonSerializer.Serialize(new { type = "transcript.text.delta", delta = Transcript })}\n\ndata: {JsonSerializer.Serialize(new { type = "transcript.text.done", text = Transcript })}\n\n", "text/event-stream"),
            (_, "" or "json") => (JsonSerializer.Serialize(new { text = Transcript }), "application/json"),
            (_, "verbose_json") => (JsonSerializer.Serialize(new { text = Transcript, language = "en", duration = 0.1,
                segments = new[] { new { id = 0, start = 0, end = 0.1, text = Transcript } } }), "application/json"),
            (_, "text") => (Transcript, "text/plain"),
            (_, "srt") => ($"1\n00:00:00,000 --> 00:00:00,100\n{Transcript}\n", "text/plain"),
            (_, "vtt") => ($"WEBVTT\n\n00:00:00.000 --> 00:00:00.100\n{Transcript}\n", "text/vtt"),
            _ => throw new ProviderRequestException("Unsupported Echo transcription response format.")
        };
        return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(body, Encoding.UTF8, type) });
    }

    public Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        CheckModel(request.Model, EmbeddingModel, ct);
        var dimensions = request.Dimensions ?? 8;
        if (dimensions is < 1 or > 1024) throw new ProviderRequestException("Echo embedding dimensions must be between 1 and 1024.");
        if (request.EncodingFormat is not (null or "float" or "base64")) throw new ProviderRequestException("Unsupported embedding encoding.");
        JsonElement[] inputs = request.Input.ValueKind switch
        {
            JsonValueKind.String => [request.Input],
            JsonValueKind.Array when request.Input.GetArrayLength() > 0 =>
                request.Input[0].ValueKind == JsonValueKind.Number ? [request.Input] : [.. request.Input.EnumerateArray()],
            _ => throw new ProviderRequestException("Expected embedding text, text array, or token input.")
        };
        if ((long)inputs.Length * dimensions > 1_000_000) throw new ProviderRequestException("Echo embedding response is too large.", 413);
        return Task.FromResult(new OpenAIEmbeddingResponse { Model = request.Model, Data = inputs.Select((input, index) =>
        {
            ct.ThrowIfCancellationRequested();
            var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(input.ValueKind == JsonValueKind.String ? input.GetString()! : input.GetRawText()));
            var vector = Enumerable.Range(0, dimensions).Select(i => bytes[i % bytes.Length] / 127.5f - 1f).ToArray();
            if (request.EncodingFormat != "base64") return new OpenAIEmbeddingData { Index = index, Embedding = JsonSerializer.SerializeToElement(vector) };
            var packed = new byte[4 * dimensions];
            for (var i = 0; i < dimensions; i++) BinaryPrimitives.WriteSingleLittleEndian(packed.AsSpan(4 * i), vector[i]);
            return new OpenAIEmbeddingData { Index = index, Embedding = JsonSerializer.SerializeToElement(Convert.ToBase64String(packed)) };
        }).ToArray() });
    }

    public Task<RerankingResponse> RerankAsync(RerankingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        CheckModel(request.Model, RerankModel, ct);
        if (request.Documents?.Values.ValueKind != JsonValueKind.Array || string.IsNullOrWhiteSpace(request.Query))
            throw new ProviderRequestException("Expected a query and documents array.");
        return Task.FromResult(new RerankingResponse { Response = new() { ModelId = "echo/" + request.Model },
            Ranking = request.Documents.Values.EnumerateArray().Select((doc, index) => new RerankingRanking { Index = index,
                RelevanceScore = (doc.ValueKind == JsonValueKind.String ? doc.GetString()! : doc.GetRawText()).Contains(request.Query, StringComparison.OrdinalIgnoreCase) ? 1 : 0 })
                .OrderByDescending(x => x.RelevanceScore).Take(request.TopN ?? int.MaxValue).ToArray() });
    }

    public Task<RealtimeTokenResponse> CreateRealtimeTokenAsync(string model, RealtimeTokenRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        CheckModel(model, RealtimeModel, ct);
        return Task.FromResult(new RealtimeTokenResponse { Token = FixtureToken, Url = FixtureUrl, ExpiresAt = 0 });
    }

    public Task<RealtimeSessionResponse> CreateRealtimeSessionAsync(string model, RealtimeSessionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        CheckModel(model, RealtimeModel, ct);
        return Task.FromResult(new RealtimeSessionResponse { SessionId = FixtureToken, Sdp = RejectedSdp });
    }

    public Task<JsonElement> CreateClientSecretAsync(JsonObject request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        CheckModel(request["session"]?["model"]?.GetValue<string>(), RealtimeModel, ct);
        return Task.FromResult(JsonSerializer.SerializeToElement(new { value = FixtureToken, expires_at = 0, url = FixtureUrl,
            session = new { type = "realtime", model = RealtimeModel } }));
    }

    public Task<LiveSessionResponse> CreateLiveSessionAsync(LiveSessionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        CheckModel(request.Session.Model, RealtimeModel, ct);
        LiveResponseTransport transport = request.Transport switch
        {
            LiveWebRtcRequestTransport => new LiveWebRtcResponseTransport { Sdp = RejectedSdp },
            LiveSipRequestTransport => new LiveSipResponseTransport(),
            _ => throw new ProviderRequestException("Echo Live fixtures support WebRTC and SIP session responses.")
        };
        return Task.FromResult(new LiveSessionResponse { Session = new() { Id = FixtureToken }, Transport = transport });
    }
}
