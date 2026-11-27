using System.Globalization;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class SonioxProviderRegistration
{
    public static IServiceCollection AddSoniox(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, SonioxProvider.Id, "https://api.soniox.com/");
        ProviderServiceCollectionExtensions.AddClient(services, SonioxProvider.SpeechClient, "https://tts-rt.soniox.com/");
        return services.AddProvider<SonioxProvider>(SonioxProvider.Id);
    }
}

internal sealed partial class SonioxProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IChatAudioRoutingProvider, IStreamingSpeechProvider, ITranscriptionProvider,
        IStreamingTranscriptionProvider, IRealtimeTokenProvider
{
    internal const string Id = "soniox";
    internal const string SpeechClient = Id + "-speech";
    private static readonly HashSet<string> NoReservedOptions = new(StringComparer.Ordinal);
    private string ApiKey() => RequiredKey(credentials, Id);

    public bool IsSpeechModel(string model) => model.StartsWith("tts-", StringComparison.OrdinalIgnoreCase);
    public bool IsTranscriptionModel(string model) => model.StartsWith("stt-", StringComparison.OrdinalIgnoreCase)
        && !model.StartsWith("stt-rt-", StringComparison.OrdinalIgnoreCase);

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (credentials.GetApiKey(Id) is not { } key || string.IsNullOrWhiteSpace(key)) yield break;
        using var client = clients.CreateClient(Id);
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var item in Array(Property(await GetAsync(client, key, "v1/models", ct), "models")))
        {
            if (String(item, "id") is not { Length: > 0 } id || !seen.Add(id)) continue;
            var mode = String(item, "transcription_mode");
            var tags = LanguageTags(item).ToList();
            if (mode is not null) tags.Add(mode.Replace('_', '-'));
            if (string.Equals(mode, "real_time", StringComparison.OrdinalIgnoreCase)) tags.AddRange(["realtime", "realtime-token"]);
            yield return Model(id, String(item, "name") ?? id, "transcription", $"Soniox {mode ?? "speech-to-text"} transcription model.", tags);
        }

        foreach (var item in Array(Property(await GetAsync(client, key, "v1/tts-models", ct), "models")))
        {
            if (String(item, "id") is not { Length: > 0 } id) continue;
            var name = String(item, "name") ?? id;
            var tags = LanguageTags(item).ToArray();
            if (seen.Add(id)) yield return Model(id, name, "speech", "Soniox text-to-speech model.", tags);
            foreach (var voice in Array(Property(item, "voices")))
            {
                if (String(voice, "id") is not { Length: > 0 } voiceId || !seen.Add($"{id}/{voiceId}")) continue;
                var voiceTags = tags.Append("voice");
                if (String(voice, "gender") is { Length: > 0 } gender) voiceTags = voiceTags.Append(gender);
                yield return Model($"{id}/{voiceId}", $"{voiceId} ({name})", "speech", String(voice, "description"), voiceTags);
            }
        }

        var cursors = new HashSet<string>(StringComparer.Ordinal);
        var path = "v1/voices?limit=100";
        for (int pageNumber = 0; pageNumber < 100; pageNumber++)
        {
            var page = await GetAsync(client, key, path, ct);
            foreach (var voice in Array(Property(page, "voices")))
            {
                if (String(voice, "id") is not { Length: > 0 } voiceId) continue;
                foreach (var model in Array(Property(voice, "models")))
                    if (String(model, "model") is { Length: > 0 } id
                        && string.Equals(String(model, "status"), "ready", StringComparison.OrdinalIgnoreCase) && seen.Add($"{id}/{voiceId}"))
                        yield return Model($"{id}/{voiceId}", String(voice, "name") ?? voiceId, "speech", $"Soniox cloned voice for {id}.", ["voice"]);
            }
            if (String(page, "next_page_cursor") is not { Length: > 0 } cursor || !cursors.Add(cursor)) yield break;
            path = $"v1/voices?limit=100&cursor={Uri.EscapeDataString(cursor)}";
        }
    }

    public async Task<RealtimeTokenResponse> CreateRealtimeTokenAsync(string model, RealtimeTokenRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var options = NativeMediaOptions.Read(Id, ScopedOptions(request.SessionConfig?.ProviderOptions), NoReservedOptions);
        var expirySeconds = OptionInt(options, "expires_in_seconds", 3600) ?? 1800;
        var duration = OptionInt(options, "max_session_duration_seconds", 18000);
        string? reference = null;
        if (options.TryGetValue("client_reference_id", out var clientReference) && clientReference.ValueKind != JsonValueKind.Null)
        {
            if (clientReference.ValueKind != JsonValueKind.String || clientReference.GetString()!.Length > 256)
                throw new ProviderRequestException("Soniox client_reference_id must be a string of at most 256 characters.");
            reference = clientReference.GetString();
        }
        var payload = new { usage_type = "transcribe_websocket", expires_in_seconds = expirySeconds,
            client_reference_id = reference, single_use = true, max_session_duration_seconds = duration };
        using var client = clients.CreateClient(Id);
        var raw = await JsonAsync<JsonElement>(client, HttpMethod.Post, "v1/auth/temporary-api-key", JsonContent.Create(payload, options: Json),
            "Bearer", ApiKey(), headers, ct);
        if (String(raw, "api_key") is not { Length: > 0 } token) throw new JsonException("Soniox returned no temporary API key.");
        var expires = DateTimeOffset.TryParse(String(raw, "expires_at"), CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var expiry)
            ? expiry : DateTimeOffset.UtcNow.AddSeconds(expirySeconds);
        return new() { Token = token, Url = "wss://stt-rt.soniox.com/transcribe-websocket", ExpiresAt = expires.ToUnixTimeSeconds() };
    }

    private static int? OptionInt(Dictionary<string, JsonElement> options, string name, int max)
    {
        if (!options.TryGetValue(name, out var value) || value.ValueKind == JsonValueKind.Null) return null;
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetInt32(out var number) || number < 1 || number > max)
            throw new ProviderRequestException($"Soniox {name} must be an integer between 1 and {max}.");
        return number;
    }

    private static Task<JsonElement> GetAsync(HttpClient client, string key, string path, CancellationToken ct)
        => NativeHttp.GetAsync<JsonElement>(client, path, headers => headers.Authorization = new("Bearer", key), ct);

    private static Model Model(string id, string name, string type, string? description, IEnumerable<string> tags) => new()
        { Id = id, Name = name, Type = type, Description = description, OwnedBy = "Soniox", Tags = tags.Distinct(StringComparer.OrdinalIgnoreCase).ToArray() };

    private static IEnumerable<string> LanguageTags(JsonElement item) => Array(Property(item, "languages"))
        .Select(language => String(language, "code")).OfType<string>().Where(code => code.Length > 0);

    private static Dictionary<string, JsonElement>? ScopedOptions(Dictionary<string, JsonElement>? options)
        => options?.TryGetValue(Id, out var native) == true ? new() { [Id] = native } : null;
}
