using System.Runtime.CompilerServices;
using System.Text.Json;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Mistral;

internal sealed partial class MistralProvider
{
    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var key = ConfiguredApiKey();
        if (string.IsNullOrWhiteSpace(key)) yield break;
        var result = await FetchModelsAsync(key, ct);
        if (Property(result, "data") is not { ValueKind: JsonValueKind.Array } data) throw new JsonException("Mistral did not return a model list.");
        var speech = new List<Model>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var item in data.EnumerateArray())
        {
            if (String(item, "id") is not { Length: > 0 } id || id.StartsWith("agent/", StringComparison.OrdinalIgnoreCase)
                || Property(item, "active").ValueKind == JsonValueKind.False || !seen.Add(id)) continue;
            var realtime = IsRealtimeSpeech(id) || IsRealtimeTranscription(id);
            var type = IsSpeechModel(id) ? "speech" : IsTranscriptionModel(id) || IsRealtimeTranscription(id) ? "transcription"
                : id.Contains("embed", StringComparison.OrdinalIgnoreCase) ? "embedding" : "language";
            var model = new Model { Id = id, Name = String(item, "name") ?? id, OwnedBy = "Mistral", Type = type,
                Description = String(item, "description"), Tags = realtime ? ["realtime", "realtime-token"] : null,
                Created = AudioCount(item, "created"), ContextWindow = Property(item, "max_context_length") is { ValueKind: JsonValueKind.Number } context
                    && context.TryGetInt32(out var size) ? size : null };
            if (type == "speech") speech.Add(model);
            yield return model;
        }
        if (speech.Count == 0) yield break;
        await foreach (var voice in VoiceCatalogAsync(ct))
            foreach (var model in speech)
            {
                var id = $"{model.Id}/{voice.Slug}";
                if (seen.Add(id)) yield return new() { Id = id, Name = id, OwnedBy = "Mistral", Type = "speech", Tags = ["voice"],
                    Description = $"{model.Name} with the {voice.Slug} voice preset." };
            }
    }

    private async Task<Voice?> FindVoiceAsync(string slug, CancellationToken ct)
    {
        await foreach (var voice in VoiceCatalogAsync(ct))
            if (string.Equals(voice.Slug, slug, StringComparison.OrdinalIgnoreCase)) return voice;
        return null;
    }

    private async IAsyncEnumerable<Voice> VoiceCatalogAsync([EnumeratorCancellation] CancellationToken ct)
    {
        const int pageSize = 100;
        using var client = CreateClient();
        var auth = Bearer(ApiKey());
        for (var offset = 0; ;)
        {
            using var request = new HttpRequestMessage(HttpMethod.Get, $"audio/voices?limit={pageSize}&offset={offset}");
            auth(request.Headers);
            using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
            if (!response.IsSuccessStatusCode) yield break; // Voice listing permissions can differ from model/synthesis permissions.
            var root = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
            if (Property(root, "items") is not { ValueKind: JsonValueKind.Array } items || items.GetArrayLength() == 0) yield break;
            foreach (var item in items.EnumerateArray())
                if (String(item, "id") is { Length: > 0 } id && String(item, "slug") is { Length: > 0 } slug
                    && slug.All(ch => char.IsAsciiLetterOrDigit(ch) || ch is '-' or '_' or '.')) yield return new(id, slug);
            offset += items.GetArrayLength();
            if (items.GetArrayLength() < pageSize || AudioCount(root, "total") is > 0 and var total && offset >= total) yield break;
        }
    }

    private sealed record Voice(string Id, string Slug);
}
