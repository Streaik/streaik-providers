using System.Globalization;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Streaik.OpenAI.Model.Providers.OpenAI.Live;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class FishAudioProvider
{
    private static readonly HashSet<string> SessionFields = new(StringComparer.Ordinal)
    {
        "agent_id", "name", "timezone", "client_timezone", "context", "overrides", "dynamic_variables", "end_user_id",
        "metadata", "llm_extra_body", "tool_events", "record_audio"
    };

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var apiKey = credentials.GetApiKey(Id);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        await foreach (var model in StaticModelListing.ListAsync(Id, apiKey, ct))
        {
            seen.Add(model.Id);
            yield return model;
        }
        if (string.IsNullOrWhiteSpace(apiKey)) yield break;
        using var client = clients.CreateClient(Id);
        var cursors = new HashSet<string>(StringComparer.Ordinal);
        string? cursor = null;
        do
        {
            var path = "v1/agent/agents?include_archived=false&publication_state=live&page_size=100";
            if (cursor is not null) path += "&cursor=" + Uri.EscapeDataString(cursor);
            using var message = Request(HttpMethod.Get, path, new Dictionary<string, string>());
            using var response = await client.SendAsync(message, HttpCompletionOption.ResponseHeadersRead, ct);
            // Agent discovery is optional for speech-only accounts.
            if (!response.IsSuccessStatusCode) yield break;
            var page = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
            var agents = Property(page, "agents");
            if (agents.ValueKind != JsonValueKind.Array) throw new JsonException("Fish Audio returned an invalid agent page.");
            foreach (var agent in agents.EnumerateArray())
            {
                if (String(agent, "status") != "active" || String(agent, "publication_state") != "live"
                    || String(agent, "agent_id") is not { Length: > 0 } id || !ValidAgentId(id) || !seen.Add($"agents/{id}")) continue;
                var tags = new List<string> { "agent", "realtime", "realtime-token", "dynamic", "livekit", "active", "live" };
                foreach (var key in new[] { "workspace_id", "updated_at" })
                    if (String(agent, key) is { } value) tags.Add($"{key}:{value}");
                if (Property(agent, "public_enabled").ValueKind == JsonValueKind.True) tags.Add("public");
                foreach (var key in new[] { "allowed_origins", "overrides_allowed" })
                    tags.AddRange(Array(Property(agent, key)).Where(v => v.ValueKind == JsonValueKind.String).Select(v => $"{key}:{v.GetString()}"));
                yield return new() { Id = $"agents/{id}", Name = String(agent, "name") ?? id, OwnedBy = "FishAudio", Type = "audio", Tags = tags,
                    Description = String(agent, "description"), Created = TryDate(String(agent, "created_at"), out var created) ? created.ToUnixTimeSeconds() : null };
            }
            if (Property(page, "has_more").ValueKind != JsonValueKind.True) yield break;
            cursor = String(page, "next_cursor");
            if (string.IsNullOrWhiteSpace(cursor) || !cursors.Add(cursor) || cursors.Count >= 1000)
                throw new JsonException("Fish Audio agent pagination did not advance.");
        } while (true);
    }

    public async Task<RealtimeTokenResponse> CreateRealtimeTokenAsync(string model, RealtimeTokenRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var config = request.SessionConfig;
        if (config is { Tools.Count: > 0 } || config?.HasTurnDetection == true || config?.InputAudioFormat is not null
            || config?.OutputAudioFormat is not null || config?.InputAudioTranscription is not null || config?.OutputAudioTranscription is not null
            || config?.OutputModalities?.Any(v => v != "audio") == true)
            throw new ProviderRequestException("Fish Audio realtime settings are configured on the published agent.", 422);
        var options = Options(config?.ProviderOptions);
        Override(options, config?.Instructions, config?.Voice);
        var session = await CreateSessionAsync(model, options, headers, ct);
        // The Vercel token contract has no metadata field. The Live transport preserves the complete join data.
        return new() { Token = Required(session, "token"), Url = Required(session, "livekit_url"), ExpiresAt = Expiry(session).ToUnixTimeSeconds() };
    }

    public async Task<LiveSessionResponse> CreateLiveSessionAsync(LiveSessionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Transport.Type != "livekit") throw new ProviderRequestException("Fish Audio requires the livekit transport.", 422);
        var config = request.Session;
        if (config.Delegation is not null || config.Client is not null || config.Input is not null || config.Triggers is not null || config.Store == true)
            throw new ProviderRequestException("Fish Audio history, tools, and delegation are configured on the published agent.", 422);
        var options = Options(config.AdditionalProperties, flat: true);
        var voice = config.Audio?.Output?.Voice;
        Override(options, config.Instructions, voice?.Name ?? voice?.Custom?.Id);
        var session = await CreateSessionAsync(config.Model, options, headers, ct);
        return new() { Session = new() { Id = Required(session, "session_id") }, Transport = new LiveAdditionalResponseTransport("livekit")
        {
            AdditionalProperties = session.EnumerateObject().Where(p => p.Name is not ("type" or "transport" or "session_id"))
                .ToDictionary(p => p.Name, p => p.Value)
        } };
    }

    private static void Override(Dictionary<string, JsonElement> options, string? instructions, string? voice)
    {
        if (instructions is null && voice is null) return;
        var raw = options.GetValueOrDefault("overrides");
        if (raw.ValueKind is not (JsonValueKind.Object or JsonValueKind.Undefined or JsonValueKind.Null))
            throw new ProviderRequestException("Fish Audio overrides must be an object.");
        var overrides = raw.ValueKind == JsonValueKind.Object ? raw.EnumerateObject().ToDictionary(p => p.Name, p => p.Value) : new();
        if (instructions is not null) overrides["system_prompt"] = JsonSerializer.SerializeToElement(instructions);
        if (voice is not null) overrides["voice_id"] = JsonSerializer.SerializeToElement(voice);
        options["overrides"] = JsonSerializer.SerializeToElement(overrides);
    }

    private async Task<JsonElement> CreateSessionAsync(string model, Dictionary<string, JsonElement> options,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var id = model.StartsWith("agents/", StringComparison.Ordinal) ? model[7..] : model;
        if (!ValidAgentId(id) || IsSpeechModel(id) || IsTranscriptionModel(id))
            throw new ProviderRequestException("A Fish Audio agent ID is required, not a speech or transcription model.");
        if (options.Keys.Any(key => !SessionFields.Contains(key))) throw new ProviderRequestException("Unsupported Fish Audio session option.");
        if (options.TryGetValue("agent_id", out var selected) && (selected.ValueKind != JsonValueKind.String || selected.GetString() != id))
            throw new ProviderRequestException("Fish Audio agent_id must match the selected model.");
        options["agent_id"] = JsonSerializer.SerializeToElement(id);
        using var client = clients.CreateClient(Id);
        using var message = Request(HttpMethod.Post, "v1/agent/sessions", headers);
        message.Content = JsonContent.Create(options, options: Json);
        using var response = await NativeHttp.SendAsync(client, message, ct);
        var session = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        _ = Required(session, "session_id");
        _ = Required(session, "token");
        if (Required(session, "transport") != "livekit" || !Uri.TryCreate(Required(session, "livekit_url"), UriKind.Absolute, out var url)
            || url.Scheme != "wss" || url.UserInfo.Length != 0)
            throw new JsonException("Fish Audio returned an invalid LiveKit transport.");
        if (Expiry(session) <= DateTimeOffset.UtcNow || Property(session, "max_duration_seconds") is not { ValueKind: JsonValueKind.Number } duration
            || !duration.TryGetInt32(out var seconds) || seconds <= 0)
            throw new JsonException("Fish Audio returned an invalid session deadline or duration.");
        return session;
    }

    private static bool ValidAgentId(string id) => id.Length is > 0 and <= 256 && id.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_' or '.');
    private static bool TryDate(string? value, out DateTimeOffset date)
        => DateTimeOffset.TryParse(value, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out date);
    private static DateTimeOffset Expiry(JsonElement session) => TryDate(Required(session, "expires_at"), out var date) ? date
        : throw new JsonException("Fish Audio returned an invalid join deadline.");
    private static string Required(JsonElement root, string field) => String(root, field) is { } value && !string.IsNullOrWhiteSpace(value)
        ? value : throw new JsonException($"Fish Audio session is missing {field}.");
}
