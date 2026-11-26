using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class GoogleGeminiProvider(IHttpClientFactory clients, IProviderCredentials credentials, ILogger<GoogleGeminiProvider> logger)
    : CompatibleChatProvider(clients, credentials, "google", "v1beta/openai/"), IRealtimeTokenProvider
{
    private readonly ILogger<GoogleGeminiProvider> _logger = logger;

    internal static bool IsLiveTranscriptionModel(string model) => model == "gemini-3.5-transcribe-live";

    internal static bool IsLiveModel(string model) => model.StartsWith("gemini-", StringComparison.Ordinal) && !model.Contains('/')
        && (model.Contains("-live", StringComparison.Ordinal) || model.Contains("-native-audio", StringComparison.Ordinal));

    public async Task<RealtimeTokenResponse> CreateRealtimeTokenAsync(string model, RealtimeTokenRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        RealtimeOptions.ValidateNativeConfig(request.SessionConfig);
        var native = model.StartsWith("models/", StringComparison.Ordinal) ? model : "models/" + model;
        if (!IsLiveModel(native[7..]))
            throw new ProviderRequestException("Use a Gemini Live model.", 400);
        var payload = RealtimeOptions.Read(request, "google");
        if (!payload.TryGetPropertyValue("bidiGenerateContentSetup", out var setup)) payload["bidiGenerateContentSetup"] = setup = new JsonObject();
        if (setup is not JsonObject live) throw new ProviderRequestException("Gemini Live setup must be an object.", 400);
        live["model"] = native;
        var now = DateTimeOffset.UtcNow;
        payload["uses"] ??= 1;
        payload["expireTime"] ??= now.AddMinutes(15).ToString("O", CultureInfo.InvariantCulture);
        payload["newSessionExpireTime"] ??= now.AddMinutes(1).ToString("O", CultureInfo.InvariantCulture);
        var requestedExpiry = RealtimeOptions.ReadDate(payload["expireTime"]?.ToString());
        if (!requestedExpiry.HasValue || requestedExpiry <= now || requestedExpiry > now.AddMinutes(30))
            throw new ProviderRequestException("Gemini Live expireTime must be within the next 30 minutes.", 400);
        var key = ApiKey();
        using var client = CreateClient();
        // Live Transcribe is documented on v1beta. Leave the existing voice-model routes on v1alpha.
        var version = IsLiveTranscriptionModel(native[7..]) ? "v1beta" : "v1alpha";
        var result = await NativeHttp.PostAsync<JsonObject, JsonElement>(client, $"{version}/auth_tokens", payload,
            outgoing => outgoing.Add("x-goog-api-key", key), headers, Json, ct);
        return new() { Token = RealtimeOptions.Required(result, "name", key),
            ExpiresAt = (RealtimeOptions.ReadDate(String(result, "expireTime")) ?? requestedExpiry)?.ToUnixTimeSeconds(),
            Url = $"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.{version}.GenerativeService.BidiGenerateContentConstrained" };
    }
}

internal sealed class XaiProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, "xai", "v1/"), IRealtimeTokenProvider
{
    internal const string RealtimeModel = "grok-voice-latest";

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey("xai"))) yield break;
        var voiceListed = false;
        await foreach (var model in base.ListModelsAsync(ct))
        {
            if (model.Id == RealtimeModel) voiceListed = true;
            yield return model;
        }
        // The Voice Agent API has a fixed alias that the language-model directory may omit.
        ct.ThrowIfCancellationRequested();
        if (!voiceListed) yield return new() { Id = RealtimeModel, Name = "Grok Voice", Type = "audio", OwnedBy = "xai", Tags = ["realtime"] };
    }

    public async Task<RealtimeTokenResponse> CreateRealtimeTokenAsync(string model, RealtimeTokenRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (model != RealtimeModel) throw new ProviderRequestException("Use xai/grok-voice-latest for realtime tokens.", 400);
        RealtimeOptions.ValidateNativeConfig(request.SessionConfig, allowTools: true);
        var payload = RealtimeOptions.Read(request, "xai");
        if (payload.Any(field => field.Key != "expires_after")) throw new ProviderRequestException("xAI token options support expires_after only.", 400);
        var key = ApiKey();
        using var client = CreateClient();
        var result = await NativeHttp.PostAsync<JsonObject, JsonElement>(client, "v1/realtime/client_secrets", payload,
            outgoing => outgoing.Authorization = new("Bearer", key), headers, Json, ct);
        var expiry = Property(result, "expires_at");
        return new() { Token = RealtimeOptions.Required(result, "value", key),
            Url = "wss://api.x.ai/v1/realtime?model=" + RealtimeModel,
            ExpiresAt = expiry.ValueKind == JsonValueKind.Number && expiry.TryGetInt64(out var epoch) ? epoch : null,
            Tools = request.SessionConfig?.Tools };
    }
}

public static class AssemblyAIRegistration
{
    public static IServiceCollection AddAssemblyAI(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, "assemblyai", "https://streaming.assemblyai.com/");
        return services.AddProvider<AssemblyAIProvider>("assemblyai");
    }
}

internal sealed class AssemblyAIProvider(IHttpClientFactory clients, IProviderCredentials credentials) : IRealtimeTokenProvider, IModelListingProvider
{
    private const string Id = "assemblyai";
    private static readonly string[] Models = ["assemblyai-stt-3-7-realtime", "universal-streaming-english", "universal-streaming-multilingual"];

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        foreach (var model in Models)
        {
            ct.ThrowIfCancellationRequested();
            yield return new() { Id = model, Name = model, Type = "transcription", OwnedBy = Id, Tags = ["realtime"] };
        }
    }

    public async Task<RealtimeTokenResponse> CreateRealtimeTokenAsync(string model, RealtimeTokenRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!Models.Contains(model, StringComparer.Ordinal)) throw new ProviderRequestException("Unsupported AssemblyAI streaming model.", 400);
        RealtimeOptions.ValidateNativeConfig(request.SessionConfig);
        var options = RealtimeOptions.Read(request, Id);
        if (options.Any(field => field.Key != "expires_in_seconds"))
            throw new ProviderRequestException("AssemblyAI token options support expires_in_seconds only.", 400);
        var seconds = 60;
        if (options.TryGetPropertyValue("expires_in_seconds", out var expiry) &&
            (expiry is not JsonValue value || !value.TryGetValue<int>(out seconds) || seconds is < 1 or > 600))
            throw new ProviderRequestException("expires_in_seconds must be between 1 and 600.", 400);
        var key = RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        var result = await NativeHttp.GetAsync<JsonElement>(client, $"v3/token?expires_in_seconds={seconds}",
            outgoing => outgoing.TryAddWithoutValidation("Authorization", key), ct);
        return new() { Token = RealtimeOptions.Required(result, "token", key), ExpiresAt = DateTimeOffset.UtcNow.AddSeconds(seconds).ToUnixTimeSeconds(),
            Url = $"wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&speech_model={Uri.EscapeDataString(model)}" };
    }
}

internal static class RealtimeOptions
{
    internal static JsonObject Read(RealtimeTokenRequest request, string provider)
    {
        if (request.SessionConfig?.ProviderOptions?.TryGetValue(provider, out var value) != true) return new();
        return Copy(value) as JsonObject ?? throw new ProviderRequestException($"{provider} token options must be an object.", 400);
    }

    internal static void ValidateNativeConfig(RealtimeSessionConfig? config, bool allowTools = false)
    {
        if (config is not null && (config.Instructions is not null || config.Voice is not null || config.OutputModalities is not null
            || config.InputAudioFormat is not null || config.OutputAudioFormat is not null || config.InputAudioTranscription is not null
            || config.OutputAudioTranscription is not null || config.HasTurnDetection || (!allowTools && config.Tools is not null)))
            throw new ProviderRequestException("This token endpoint requires provider-native session options or client-side session setup.", 400);
    }

    internal static string Required(JsonElement result, string field, string key)
        => String(result, field) is { Length: > 0 } token && token != key ? token : throw new JsonException("Unexpected realtime token response.");

    internal static DateTimeOffset? ReadDate(string? value)
        => DateTimeOffset.TryParse(value, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var time) ? time : null;
}
