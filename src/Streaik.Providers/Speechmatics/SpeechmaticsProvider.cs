using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Speechmatics;

public static class SpeechmaticsProviderRegistration
{
    public static IServiceCollection AddSpeechmatics(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, SpeechmaticsProvider.Id, "https://eu1.asr.api.speechmatics.com/");
        return services.AddProvider<SpeechmaticsProvider>(SpeechmaticsProvider.Id);
    }
}

internal sealed partial class SpeechmaticsProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, ISpeechProvider, IStreamingSpeechEventsProvider, IChatAudioRoutingProvider,
      ITranscriptionProvider, INativeAudioTranscriptionProvider, IRealtimeTokenProvider
{
    internal const string Id = "speechmatics";
    private const string VoiceSegment = "tts/voice/";
    private const string AgentModel = "linden-1";
    private static readonly (string Id, string Name)[] Voices =
    [
        ("sarah", "Sarah: English Female (UK)"), ("theo", "Theo: English Male (UK)"),
        ("megan", "Megan: English Female (US)"), ("jack", "Jack: English Male (US)")
    ];
    private static readonly (string Id, string Name, string Description)[] TranscriptionModels =
    [
        ("standard", "Speechmatics Standard", "Speech-to-text transcription model"),
        ("enhanced", "Speechmatics Enhanced", "Enhanced speech-to-text transcription model"),
        ("melia-1", "Speechmatics Melia 1", "Multilingual batch and realtime transcription with diarization and word timings"),
        ("oak-1", "Speechmatics Oak 1", "Multilingual batch and realtime transcription with optional language hints"),
        (AgentModel, "Speechmatics Linden 1 (Agent STT)", "Agent STT WebSocket-only transcription; not a batch model")
    ];

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        await Task.CompletedTask;
        ct.ThrowIfCancellationRequested();
        yield return new() { Id = "tts", Name = "Speechmatics Text to Speech", Type = "speech", Description = "Provide the voice request property" };
        foreach (var (id, name) in Voices)
            yield return new() { Id = VoiceSegment + id, Name = name, Type = "speech", Tags = ["voice"], Description = "Text-to-speech voice" };
        foreach (var (id, name, description) in TranscriptionModels)
            yield return new() { Id = id, Name = name, Type = "transcription", Description = description, Tags = ["realtime"] };
    }

    public bool IsSpeechModel(string model) => model == "tts" || model.StartsWith(VoiceSegment, StringComparison.Ordinal);
    public bool IsTranscriptionModel(string model) => TranscriptionModels.Any(entry => entry.Id == model);

    public async Task<RealtimeTokenResponse> CreateRealtimeTokenAsync(string model, RealtimeTokenRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!IsTranscriptionModel(model)) throw new ProviderRequestException("Select a Speechmatics realtime transcription model.");
        var config = request.SessionConfig;
        if (config is not null && (config.Instructions is not null || config.Voice is not null || config.OutputModalities is not null
            || config.InputAudioFormat is not null || config.OutputAudioFormat is not null || config.InputAudioTranscription is not null
            || config.OutputAudioTranscription is not null || config.HasTurnDetection || config.Tools is not null))
            throw new ProviderRequestException("Speechmatics recognition settings must be sent over the client WebSocket.");
        var options = config?.ProviderOptions?.GetValueOrDefault(Id) ?? default;
        if (options.ValueKind is not (JsonValueKind.Object or JsonValueKind.Undefined or JsonValueKind.Null))
            throw new ProviderRequestException("Speechmatics realtime options must be an object.");
        var regionOption = Property(options, "region");
        var region = regionOption.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null ? "global" : String(options, "region");
        if (region is not ("global" or "eu" or "us" or "au"))
            throw new ProviderRequestException("Speechmatics region must be global, eu, us, or au.");
        var ttl = 60;
        var ttlOption = Property(options, "ttl");
        if (ttlOption.ValueKind is not (JsonValueKind.Undefined or JsonValueKind.Null)
            && (ttlOption.ValueKind != JsonValueKind.Number || !ttlOption.TryGetInt32(out ttl) || ttl is < 60 or > 86400))
            throw new ProviderRequestException("Speechmatics ttl must be an integer from 60 to 86400 seconds.");

        var key = RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        var expiresAt = DateTimeOffset.UtcNow.AddSeconds(ttl).ToUnixTimeSeconds();
        var result = await NativeHttp.PostAsync<object, JsonElement>(client, "https://mp.speechmatics.com/v1/jwts?type=rt",
            new { ttl }, outbound => outbound.Authorization = new AuthenticationHeaderValue("Bearer", key), headers, Json, ct);
        var token = String(result, "key_value");
        if (string.IsNullOrWhiteSpace(token) || token.Length > 64000 || token.Any(char.IsWhiteSpace) || token == key
            || expiresAt <= DateTimeOffset.UtcNow.ToUnixTimeSeconds())
            throw new HttpRequestException("Speechmatics returned an invalid or expired temporary key.");
        return new() { Token = token, ExpiresAt = expiresAt, Url = $"wss://{region}.rt.speechmatics.com/v2" + (model == AgentModel ? "/agent" : "") };
    }

    // Native options stay as JSON, but must not override routing, credentials, or normalized inputs.
    private static Dictionary<string, JsonElement> Options(IReadOnlyDictionary<string, JsonElement>? source, params string[] controls)
    {
        var reserved = new HashSet<string>(controls, StringComparer.OrdinalIgnoreCase)
            { Id, "providerOptions", "provider_options", "providerMetadata", "authorization", "api_key", "apiKey", "x-api-key", "headers" };
        var result = new Dictionary<string, JsonElement>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, value) in source ?? new Dictionary<string, JsonElement>())
            if (!reserved.Contains(name)) result[name] = value;
        foreach (var scoped in new[] { Property(Field("provider_options"), Id), Property(Field("providerOptions"), Id), Field(Id) })
        {
            if (scoped.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null) continue;
            if (scoped.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Speechmatics provider options must be an object.");
            foreach (var property in scoped.EnumerateObject())
                if (!reserved.Contains(property.Name)) result[property.Name] = property.Value;
        }
        return result;

        JsonElement Field(string name) => source?.TryGetValue(name, out var value) == true ? value : default;
    }
}
