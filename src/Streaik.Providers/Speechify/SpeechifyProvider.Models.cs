using System.Runtime.CompilerServices;
using System.Text.Json;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class SpeechifyProvider
{
    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        using var client = clients.CreateClient(Id);
        var auth = Bearer();
        var catalog = await NativeHttp.GetAsync<JsonElement>(client, "v1/audio/models", auth, ct);
        var models = new Dictionary<string, Model>(StringComparer.OrdinalIgnoreCase);
        foreach (var item in Items(catalog, "models", "data"))
        {
            if (Clean(item, "id") is not { } id) continue;
            var tags = Languages(item).ToList();
            if (Property(item, "default").ValueKind == JsonValueKind.True) tags.Add("default");
            if (Property(item, "recommended").ValueKind == JsonValueKind.True) tags.Add("recommended");
            var model = new Model { Id = id, Name = Clean(item, "name") ?? id, Description = Clean(item, "description"),
                Type = "speech", OwnedBy = Id, Tags = tags };
            if (models.TryAdd(id, model)) yield return model;
        }

        string? cursor = null;
        var seen = new HashSet<string>(models.Keys, StringComparer.OrdinalIgnoreCase);
        var seenCursors = new HashSet<string>(StringComparer.Ordinal);
        do
        {
            var path = cursor is null ? "v1/voices?limit=200" : "v1/voices?limit=200&cursor=" + Uri.EscapeDataString(cursor);
            var result = await NativeHttp.GetAsync<JsonElement>(client, path, auth, ct);
            foreach (var voice in Items(result, "voices", "data", "items", "results"))
            {
                var voiceId = Clean(voice, "id") ?? Clean(voice, "voice_id") ?? Clean(voice, "voiceId");
                if (voiceId is null || voiceId.Contains('/')) continue;
                var name = Clean(voice, "display_name") ?? Clean(voice, "displayName") ?? Clean(voice, "name") ?? voiceId;
                foreach (var available in Array(Property(voice, "models")))
                {
                    if (Clean(available, "name") is not { } modelId || modelId.Contains('/') || !seen.Add($"{modelId}/{voiceId}")) continue;
                    var tags = new[] { "voice", modelId, Clean(voice, "gender"), Clean(voice, "locale"), Clean(voice, "type") }
                        .OfType<string>().Concat(Languages(available)).Concat(Array(Property(voice, "tags"))
                        .Where(tag => tag.ValueKind == JsonValueKind.String).Select(tag => tag.GetString()!)).Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
                    var description = models.GetValueOrDefault(modelId)?.Description;
                    yield return new() { Id = $"{modelId}/{voiceId}", Name = $"{name} ({modelId})", Type = "speech", OwnedBy = Id,
                        Description = description is null ? $"{name} voice for {modelId}." : $"{name} voice for {modelId}. {description}", Tags = tags };
                }
            }
            cursor = Property(result, "has_more").ValueKind == JsonValueKind.True ? Clean(result, "next_cursor") : null;
        } while (cursor is not null && seenCursors.Add(cursor));
    }

    private static IEnumerable<JsonElement> Items(JsonElement body, params string[] names)
        => Array(body.ValueKind == JsonValueKind.Array ? body : names.Select(name => Property(body, name)).FirstOrDefault(e => e.ValueKind == JsonValueKind.Array));

    private static IEnumerable<string> Languages(JsonElement item) => Array(Property(item, "languages"))
        .Select(value => value.ValueKind == JsonValueKind.String ? value.GetString() : Clean(value, "locale"))
        .Where(value => !string.IsNullOrWhiteSpace(value)).Select(value => value!.Trim()).Distinct(StringComparer.OrdinalIgnoreCase);

    private static string? Clean(JsonElement value, string name) => String(value, name)?.Trim() is { Length: > 0 } text ? text : null;
}
