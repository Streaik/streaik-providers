using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class ElevenLabsRegistration
{
    public static IServiceCollection AddElevenLabs(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, "elevenlabs", "https://api.elevenlabs.io/v1/");
        return services.AddProvider<ElevenLabsProvider>("elevenlabs");
    }
}

internal sealed partial class ElevenLabsProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, ISpeechProvider, IStreamingSpeechEventsProvider, IChatAudioRoutingProvider,
      ITranscriptionProvider, INativeAudioTranscriptionProvider, IRealtimeTokenProvider
{
    private const string Id = "elevenlabs";
    private const string SoundModel = "eleven_text_to_sound_v2";
    private const string DialogueSuffix = "/text-to-dialogue";
    private static readonly string[] TranscriptionModels = ["scribe_v1", "scribe_v2", "scribe_v2_medical"];
    private static readonly string[] RealtimeModels = ["scribe_v2_realtime", "scribe_v2_realtime_turbo", "scribe_v2_realtime_lite"];

    public bool IsTranscriptionModel(string model) => TranscriptionModels.Contains(model, StringComparer.Ordinal);
    public bool IsSpeechModel(string model) => model.StartsWith("eleven_", StringComparison.Ordinal) || IsMusic(model) || IsDialogue(model);
    private static bool IsMusic(string model) => model is "music_v1" or "music_v2";
    private static bool IsDialogue(string model) => model.EndsWith(DialogueSuffix, StringComparison.Ordinal);

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var key = credentials.GetApiKey(Id);
        if (string.IsNullOrWhiteSpace(key)) yield break;
        using var client = clients.CreateClient(Id);
        var page = await NativeHttp.GetAsync<JsonElement>(client, "models", h => h.Add("xi-api-key", key), ct);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var item in Array(page))
        {
            if (String(item, "model_id") is not { Length: > 0 } model || Property(item, "can_do_text_to_speech").ValueKind != JsonValueKind.True) continue;
            var name = String(item, "name") ?? model;
            var description = String(item, "description");
            if (seen.Add(model)) yield return Entry(model, "speech", name, description);
            if (Property(item, "can_do_text_to_dialogue").ValueKind == JsonValueKind.True && seen.Add(model + DialogueSuffix))
                yield return Entry(model + DialogueSuffix, "speech", name + " Dialogue", description);
        }
        foreach (var model in new[] { SoundModel, "music_v1", "music_v2" })
            if (seen.Add(model)) yield return Entry(model, "speech");
        foreach (var model in TranscriptionModels)
            if (seen.Add(model)) yield return Entry(model, "transcription");
        foreach (var model in RealtimeModels)
            if (seen.Add(model)) yield return Entry(model, "transcription", tags: ["realtime"]);

        static Model Entry(string model, string type, string? name = null, string? description = null, string[]? tags = null)
            => new() { Id = model, OwnedBy = Id, Name = name ?? model, Type = type, Description = description, Tags = tags };
    }

    public async Task<RealtimeTokenResponse> CreateRealtimeTokenAsync(string model, RealtimeTokenRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!RealtimeModels.Contains(model, StringComparer.Ordinal)) throw new ProviderRequestException("Unsupported ElevenLabs realtime Scribe model.");
        RealtimeOptions.ValidateNativeConfig(request.SessionConfig);
        if (RealtimeOptions.Read(request, Id).Count != 0) throw new ProviderRequestException("ElevenLabs Scribe session options must be set by the realtime client.");
        var key = RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        using var response = await SendAsync(client, HttpMethod.Post, "single-use-token/realtime_scribe", JsonContent.Create(new JsonObject(), options: Json), headers, ct);
        var result = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        return new() { Token = RealtimeOptions.Required(result, "token", key), ExpiresAt = DateTimeOffset.UtcNow.AddMinutes(15).ToUnixTimeSeconds(),
            Url = $"wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id={Uri.EscapeDataString(model)}" };
    }

    private async Task<HttpResponseMessage> SendAsync(HttpClient client, HttpMethod method, string path, HttpContent? content,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct, string? accept = null)
    {
        using var request = new HttpRequestMessage(method, path) { Content = content };
        request.Headers.Add("xi-api-key", RequiredKey(credentials, Id));
        if (accept is not null) request.Headers.Accept.ParseAdd(accept);
        NativeHttp.ApplyHeaders(request, headers);
        return await NativeHttp.SendAsync(client, request, ct);
    }

    private static JsonObject Options(IReadOnlyDictionary<string, JsonElement>? source)
    {
        var result = new JsonObject(new JsonNodeOptions { PropertyNameCaseInsensitive = true });
        if (source is not null)
            foreach (var (name, value) in source)
                if (!Blocked.Contains(name)) result[name] = Copy(value);
        return result;
    }

    private static readonly HashSet<string> Blocked = new(["authorization", "api_key", "api-key", "x-api-key", "xi-api-key", "headers", "endpoint", "url"], StringComparer.OrdinalIgnoreCase);
    private static string? Text(JsonNode? value) => value is JsonValue scalar && scalar.TryGetValue<string>(out var text) ? text : null;
    private static bool Flag(JsonNode? value) => value is JsonValue scalar && (scalar.TryGetValue<bool>(out var flag) ? flag : bool.TryParse(Text(scalar), out flag) && flag);
    private static string? TakeQuery(JsonObject fields, string name)
        => fields.Remove(name, out var value) && value is JsonValue ? $"{name}={Uri.EscapeDataString(Text(value) ?? value.ToJsonString())}" : null;
    private static Dictionary<string, JsonElement> Metadata(JsonElement body) => new() { [Id] = body };
}
