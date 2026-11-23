using System.Globalization;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Streaik.Interactions.Models;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class GoogleGeminiProvider
{
    private const string InteractionsPath = "v1beta/interactions";
    internal const string InteractionRevision = "2026-05-20";

    private static Action<HttpRequestHeaders> GoogleAuth(string key, bool revision = false) => headers =>
    {
        headers.Add("x-goog-api-key", key);
        if (revision) headers.Add("Api-Revision", InteractionRevision);
    };

    private Task<HttpResponseMessage> CreateInteractionAsync(InteractionRequest request, CancellationToken ct)
        => SendGoogleAsync(HttpMethod.Post, InteractionsPath, ct, JsonContent.Create(request, options: InteractionJson.Default), request.Stream == true);

    private Task<HttpResponseMessage> GetInteractionAsync(string id, CancellationToken ct)
        => SendGoogleAsync(HttpMethod.Get, $"{InteractionsPath}/{ResourceId(id)}", ct);

    private Task<HttpResponseMessage> DeleteInteractionAsync(string id, CancellationToken ct)
        => SendGoogleAsync(HttpMethod.Delete, $"{InteractionsPath}/{ResourceId(id)}", ct);

    private Task<HttpResponseMessage> ListAgentsAsync(string? pageToken, CancellationToken ct)
        => SendGoogleAsync(HttpMethod.Get, "v1beta/agents?page_size=1000" +
            (string.IsNullOrEmpty(pageToken) ? "" : "&page_token=" + Uri.EscapeDataString(pageToken)), ct);

    private async Task<HttpResponseMessage> SendGoogleAsync(HttpMethod method, string path, CancellationToken ct,
        HttpContent? content = null, bool stream = false)
    {
        using var request = new HttpRequestMessage(method, path) { Content = content };
        GoogleAuth(ApiKey(), revision: true)(request.Headers);
        request.Headers.Accept.ParseAdd(stream ? "text/event-stream" : "application/json");
        using var client = CreateClient();
        return await NativeHttp.SendAsync(client, request, ct);
    }

    private static string ResourceId(string id)
    {
        if (id.Length is < 1 or > 512 || id is "." or ".." || id.Any(c => !(char.IsAsciiLetterOrDigit(c) || c is '-' or '_' or '.')))
            throw new ProviderRequestException("Invalid Google resource ID.");
        return id;
    }

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var key = ConfiguredApiKey();
        if (string.IsNullOrWhiteSpace(key)) yield break;
        var seenModels = new HashSet<(string, string?)>();
        var nativeModels = await ReadGoogleModelsAsync(key, ct);
        var source = nativeModels.Count > 0 ? Enumerate(nativeModels, ct) : base.ListModelsAsync(ct);
        await foreach (var model in source.WithCancellation(ct))
        {
            if (model.Id == "gemini-omni-flash-preview") continue;
            if (IsLiveTranscriptionModel(model.Id)) model.Type = "transcription";
            if (GoogleModelIds.IsAgent(model.Id) || model.Type is "agent" or "agents")
            {
                if (!GoogleModelIds.IsAgent(model.Id)) model.Id = "agents/" + model.Id;
                model.Type = "language";
                model.Tags = [.. (model.Tags ?? []).Where(t => !t.Equals("agent", StringComparison.OrdinalIgnoreCase)
                    && !t.Equals("agents", StringComparison.OrdinalIgnoreCase)), "agent"];
            }
            if (seenModels.Add((model.Id, model.Type))) yield return model;
            if (model.Id.Contains("-omni-", StringComparison.OrdinalIgnoreCase) && seenModels.Add((model.Id, "video")))
                yield return new Model { Id = model.Id, Name = model.Name + " Video", OwnedBy = model.OwnedBy, Type = "video" };
        }

        var seen = new HashSet<string>(StringComparer.Ordinal);
        var pages = new HashSet<string>(StringComparer.Ordinal);
        string? token = null;
        do
        {
            JsonElement page;
            try
            {
                using var response = await ListAgentsAsync(token, ct);
                page = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
            }
            catch (Exception ex) when (!ct.IsCancellationRequested && ex is ProviderCredentialException or HttpRequestException or JsonException)
            { yield break; } // Agent discovery is optional for otherwise valid Google credentials.

            var entries = Property(page, "agents");
            foreach (var item in Array(entries.ValueKind == JsonValueKind.Array ? entries : Property(page, "data")))
            {
                if (String(item, "id") is not { Length: > 0 } id || id.Length > 512 || id is "." or ".."
                    || id.Any(c => !(char.IsAsciiLetterOrDigit(c) || c is '-' or '_' or '.')) || !seen.Add(id)
                    || !seenModels.Add(($"agents/{id}", "language"))) continue;
                yield return new Model { Id = $"agents/{id}", Name = String(item, "display_name") ?? id,
                    Description = String(item, "description"), OwnedBy = "google", Type = "language", Tags = ["agent"],
                    Created = DateTimeOffset.TryParse(String(item, "created"), CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var date)
                        ? date.ToUnixTimeSeconds() : null };
            }
            token = String(page, "next_page_token");
        } while (!string.IsNullOrEmpty(token) && pages.Add(token));
    }

    private async Task<List<Model>> ReadGoogleModelsAsync(string key, CancellationToken ct)
    {
        using var client = CreateClient();
        var models = new List<Model>();
        var tokens = new HashSet<string>(StringComparer.Ordinal);
        string? token = null;
        do
        {
            var path = "v1beta/models?pageSize=1000" + (token is null ? "" : "&pageToken=" + Uri.EscapeDataString(token));
            JsonElement page;
            try { page = await NativeHttp.GetAsync<JsonElement>(client, path, GoogleAuth(key), ct); }
            catch (Exception ex) when (!ct.IsCancellationRequested && ex is HttpRequestException or JsonException or OperationCanceledException) { break; }
            foreach (var item in Array(Property(page, "models")))
            {
                if (String(item, "name") is not { Length: > 0 } resource || !resource.StartsWith("models/", StringComparison.Ordinal)) continue;
                var id = resource[7..];
                if (id.Length == 0 || id.Contains('/')) continue;
                var type = GoogleModelType(id, item);
                models.Add(new() { Id = id, Name = String(item, "displayName") ?? id, Description = String(item, "description"), OwnedBy = "google",
                    Type = type, Tags = IsLiveModel(id) ? ["realtime"] : null, ContextWindow = TokenLimit(item, "inputTokenLimit"),
                    MaxTokens = TokenLimit(item, "outputTokenLimit") });
                if (type == "language" && (id.StartsWith("gemini-2", StringComparison.OrdinalIgnoreCase) || id.StartsWith("gemini-3", StringComparison.OrdinalIgnoreCase))
                    && !id.Contains("-omni-", StringComparison.OrdinalIgnoreCase))
                    models.Add(new() { Id = id, Name = (String(item, "displayName") ?? id) + " Transcription", OwnedBy = "google", Type = "transcription" });
            }
            token = String(page, "nextPageToken");
        } while (!string.IsNullOrEmpty(token) && tokens.Add(token));
        return models;
    }

    private static int? TokenLimit(JsonElement item, string key) => Property(item, key) is { ValueKind: JsonValueKind.Number } n
        && n.TryGetInt32(out var size) && size > 0 ? size : null;

    private static string GoogleModelType(string id, JsonElement entry)
    {
        if (GoogleModelIds.IsAgent(id)) return "language";
        if (IsLiveTranscriptionModel(id)) return "transcription";
        if (IsLiveModel(id)) return "audio";
        var methods = Array(Property(entry, "supportedGenerationMethods")).Where(x => x.ValueKind == JsonValueKind.String).Select(x => x.GetString()).ToHashSet();
        if (id.StartsWith("gemini-decision-", StringComparison.OrdinalIgnoreCase)) return "decision";
        if (id.StartsWith("veo-", StringComparison.OrdinalIgnoreCase)) return "video";
        if (id.Contains("embedding", StringComparison.OrdinalIgnoreCase) || methods.Contains("embedContent")) return "embedding";
        if (id.Contains("-tts", StringComparison.OrdinalIgnoreCase)) return "speech";
        if (id.StartsWith("imagen-", StringComparison.OrdinalIgnoreCase) || id.Contains("-image", StringComparison.OrdinalIgnoreCase)) return "image";
        if (id.StartsWith("gemini-", StringComparison.OrdinalIgnoreCase) && methods.Contains("generateContent")) return "language";
        return "other";
    }

    private static async IAsyncEnumerable<Model> Enumerate(List<Model> models, [EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        foreach (var model in models) { ct.ThrowIfCancellationRequested(); yield return model; }
    }

}

internal static class GoogleModelIds
{
    internal static bool IsAgent(string model) => model.StartsWith("agents/", StringComparison.OrdinalIgnoreCase)
        || model.StartsWith("deep-research", StringComparison.OrdinalIgnoreCase)
        || model.StartsWith("antigravity", StringComparison.OrdinalIgnoreCase)
        || model.StartsWith("code-mender", StringComparison.OrdinalIgnoreCase);
}
