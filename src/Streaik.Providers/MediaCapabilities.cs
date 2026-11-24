using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Streaik.OpenAI.Models;
using Streaik.OpenAI.Model.Providers.OpenAI.Live;
using Streaik.Vercel.Models;

namespace Streaik.Providers;

public interface IOpenAILiveSessionProvider
{
    Task<LiveSessionResponse> CreateLiveSessionAsync(LiveSessionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}

public interface IImageProvider
{
    Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
    Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}

// Optional dispatch for providers that offer both chat and image models.
public interface IImageModelRoutingProvider : IImageProvider
{
    Task<bool> IsImageModelAsync(string model, CancellationToken ct);
}

public interface IVideoProvider
{
    Task<VideoOperationStartResult> StartVideoAsync(VideoRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
    Task<VideoOperationStatusResult> GetVideoAsync(string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}

public interface IStreamingImagesProvider
{
    IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
    IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}

public interface ISpeechProvider
{
    // The caller owns the response and streams or buffers its audio content according to the client protocol.
    Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}

public interface IStreamingSpeechProvider : ISpeechProvider
{
    // Returns a binary audio stream. The gateway supplies client-specific framing.
    Task<HttpResponseMessage> SpeakStreamingAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}

// For upstreams that return structured audio events, not just a binary stream.
public interface IStreamingSpeechEventsProvider
{
    IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
    IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}

// Opt in to direct audio turns on chat surfaces that can represent the input and output.
public interface IChatAudioRoutingProvider
{
    bool IsSpeechModel(string model);
    bool IsTranscriptionModel(string model);
}

public interface ITranscriptionProvider
{
    Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}

public interface IStreamingTranscriptionProvider
{
    IAsyncEnumerable<StreamingTranscriptionPart> StreamTranscriptionAsync(StreamingTranscriptionRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}

public interface INativeAudioTranscriptionProvider
{
    Task<HttpResponseMessage> TranscribeNativeAsync(string model, IFormCollection request, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}

public interface IRealtimeTokenProvider
{
    Task<RealtimeTokenResponse> CreateRealtimeTokenAsync(string model, RealtimeTokenRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}

public interface IRealtimeSessionProvider
{
    Task<RealtimeSessionResponse> CreateRealtimeSessionAsync(string model, RealtimeSessionRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}

public interface IOpenAIRealtimeProvider
{
    Task<JsonElement> CreateClientSecretAsync(System.Text.Json.Nodes.JsonObject request, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}

public interface IEmbeddingsProvider
{
    Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}

public interface IEmbeddingMetadataProvider : IEmbeddingsProvider
{
    Task<(OpenAIEmbeddingResponse Result, JsonElement Body, Dictionary<string, string?> Headers)> EmbedWithMetadataAsync(
        OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}

public interface IRerankingProvider
{
    Task<RerankingResponse> RerankAsync(RerankingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}

public interface IContainerFilesProvider
{
    IAsyncEnumerable<ContainerFileEntry> ListContainerFilesAsync(string containerId,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct);
    Task<HttpResponseMessage> DownloadContainerFileAsync(string containerId, string fileId,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct);
    Task<bool> IsContainerActiveAsync(string containerId, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}

public sealed record ContainerFileEntry(string Id, string? Path, string? Source, long? CreatedAt);

public sealed class ProviderRequestException(string message, int status = 400) : Exception(message)
{
    public int Status { get; } = status;
}
