using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class RunwayProviderRegistration
{
    public static IServiceCollection AddRunway(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, RunwayProvider.Id, "https://api.dev.runwayml.com/v1/");
        PublicMediaDelivery.AddClient(services, RunwayProvider.Delivery);
        return services.AddProvider<RunwayProvider>(RunwayProvider.Id);
    }
}

internal sealed partial class RunwayProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IImageModelRoutingProvider, IStreamingImagesProvider, IStreamingSpeechProvider, IChatAudioRoutingProvider, IVideoProvider
{
    internal const string Id = "runway";
    internal const string Delivery = "runway-delivery";
    private const int InlineLimit = 24 * 1024 * 1024;
    private const string ApiVersion = "2024-11-06";
    private static readonly Lazy<Dictionary<string, string>> ModelTypes = new(() =>
        (Streaik.Catalog.ProviderCatalog.ReadModels(Id)?.Deserialize<ModelResponse>(Json)?.Data ?? [])
            .ToDictionary(model => model.Id["runway/".Length..], model => model.Type, StringComparer.OrdinalIgnoreCase));

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var key = credentials.GetApiKey(Id);
        if (string.IsNullOrWhiteSpace(key)) yield break;
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        await foreach (var model in StaticModelListing.ListAsync(Id, key, ct))
            if (seen.Add(model.Id)) yield return model;

        string? cursor = null;
        var cursors = new HashSet<string>(StringComparer.Ordinal);
        using var client = clients.CreateClient(Id);
        do
        {
            JsonElement page;
            try
            {
                page = await NativeHttp.GetAsync<JsonElement>(client, cursor is null ? "avatars" : $"avatars?cursor={Uri.EscapeDataString(cursor)}", Authenticate(), ct);
            }
            catch (HttpRequestException) { yield break; } // Custom avatars are additive; keep the static catalog.
            foreach (var avatar in Array(Property(page, "data")))
            {
                if (String(avatar, "id") is not { Length: > 0 } id || !seen.Add("avatar/" + id)) continue;
                var name = String(avatar, "name") ?? id;
                var voice = Property(avatar, "voice");
                var preset = String(voice, "name") ?? String(voice, "presetId") ?? String(voice, "id");
                var status = String(avatar, "status");
                yield return new Model { Id = $"avatar/{id}", Name = "Avatar: " + name, OwnedBy = "Runway", Type = "video", Tags = ["persona"],
                    Description = $"Runway custom avatar '{name}'{(preset is null ? "" : $" using voice '{preset}'")}{(status is null ? "" : $" ({status})") }.",
                    Created = DateTimeOffset.TryParse(String(avatar, "createdAt"), out var created) ? created.ToUnixTimeSeconds() : null };
            }
            cursor = Property(page, "hasMore").ValueKind == JsonValueKind.True ? String(page, "nextCursor") : null;
        } while (!string.IsNullOrWhiteSpace(cursor) && cursors.Add(cursor));
    }

    public Task<bool> IsImageModelAsync(string model, CancellationToken ct) => Task.FromResult(ModelTypes.Value.GetValueOrDefault(model) == "image");
    public bool IsSpeechModel(string model) => ModelTypes.Value.GetValueOrDefault(model) == "speech";
    public bool IsTranscriptionModel(string model) => false;

    private Action<HttpRequestHeaders> Authenticate()
    {
        var key = RequiredKey(credentials, Id);
        return headers => { headers.Authorization = new("Bearer", key); headers.Add("X-Runway-Version", ApiVersion); };
    }

    private async Task<(JsonElement Body, Dictionary<string, string?> Headers)> SubmitAsync(string path, JsonObject payload,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        return await NativeHttp.PostJsonAsync(client, path, payload, Authenticate(), headers, Json, ct);
    }

    private static string TaskId(JsonElement body)
        => String(body, "id") is { Length: > 0 } id && id is not ("." or "..") ? id : throw new JsonException("Runway returned no valid task ID.");

    private async Task<JsonElement> ReadTaskAsync(string id, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        return await NativeHttp.GetAsync<JsonElement>(client, $"tasks/{Uri.EscapeDataString(id)}", Authenticate(), ct);
    }

    private async Task<JsonElement> WaitAsync(string id, CancellationToken ct)
    {
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromMinutes(5));
        try
        {
            while (true)
            {
                var task = await ReadTaskAsync(id, deadline.Token);
                if (String(task, "status")?.ToUpperInvariant() == "SUCCEEDED") return task;
                if (Failure(task) is { } failure) throw new ProviderRequestException(failure, 502);
                if (String(task, "status") is null) throw new JsonException("Runway returned no task status.");
                await Task.Delay(TimeSpan.FromSeconds(1), deadline.Token);
            }
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested && deadline.IsCancellationRequested)
        { throw new ProviderRequestException("Runway task timed out.", 504); }
    }

    private static string? Failure(JsonElement task) => String(task, "status")?.ToUpperInvariant() is "FAILED" or "CANCELED" or "CANCELLED"
        ? $"Runway task failed: {String(task, "failure") ?? "Unknown failure."}{(String(task, "failureCode") is { } code ? $" ({code})" : "")}" : null;

    private static IEnumerable<string> Outputs(JsonElement task)
    {
        var output = Property(task, "output");
        if (output.ValueKind == JsonValueKind.String && output.GetString() is { Length: > 0 } direct) yield return direct;
        foreach (var value in Array(output))
        {
            var url = value.ValueKind == JsonValueKind.String ? value.GetString() : String(value, "uri") ?? String(value, "url");
            if (!string.IsNullOrWhiteSpace(url)) yield return url;
        }
    }

    private async Task<(string Data, string Mime)> DownloadAsync(string url, string fallback, int limit, CancellationToken ct)
    {
        using var delivery = clients.CreateClient(Delivery);
        return await PublicMediaDelivery.DownloadAsync(delivery, url, fallback, limit, ct);
    }

    private static JsonObject Options(IReadOnlyDictionary<string, JsonElement>? properties, params string[] reserved)
        => ToObject(NativeMediaOptions.Read(Id, properties, new HashSet<string>(reserved, StringComparer.OrdinalIgnoreCase)));
}
