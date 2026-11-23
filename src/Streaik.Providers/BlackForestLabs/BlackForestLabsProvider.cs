using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.BlackForestLabs;

public static class BlackForestLabsRegistration
{
    public static IServiceCollection AddBlackForestLabs(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, BlackForestLabsProvider.Id, "https://api.bfl.ai/");
        ProviderServiceCollectionExtensions.AddClient(services, BlackForestLabsProvider.AssetClient, "https://delivery.bfl.ai/");
        return services.AddProvider<BlackForestLabsProvider>(BlackForestLabsProvider.Id);
    }
}

internal sealed partial class BlackForestLabsProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IImageProvider, IVideoProvider
{
    internal const string Id = "blackforestlabs";
    internal const string AssetClient = "blackforestlabs-assets";
    private const int ImageLimit = 32 * 1024 * 1024;
    private const int VideoLimit = 64 * 1024 * 1024;
    private static readonly string[] ImageModels = ["flux-3-image", "flux-3-image-klein", "flux-2-max", "flux-2-pro",
        "flux-2-pro-preview", "flux-2-flex", "flux-2-klein-9b", "flux-2-klein-9b-preview", "flux-2-klein-4b",
        "flux-tools/outpainting-v1", "flux-tools/erase-v1", "flux-tools/deblur-v1"];
    private static readonly string[] VideoModels = ["flux-3-video", "flux-3-video-draft-enhance", "flux-tools/video-edit-v1", "flux-tools/video-upscale-v1"];
    private static readonly string[] ImageMediaFields = ["sample", "samples", "images", "image", "url", "b64_json", "base64"];

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        foreach (var model in ImageModels.Concat(VideoModels))
        {
            ct.ThrowIfCancellationRequested();
            yield return new() { Id = model, Name = model, OwnedBy = "BlackForestLabs", Type = ImageModels.Contains(model) ? "image" : "video" };
        }
    }

    private Action<HttpRequestHeaders> Auth()
    {
        var key = RequiredKey(credentials, Id);
        return headers => headers.Add("x-key", key);
    }

    private static (string Id, Uri Url) PollTarget(JsonElement body)
    {
        var taskId = String(body, "id");
        if (string.IsNullOrWhiteSpace(taskId)) throw new IOException("Black Forest Labs did not return a task ID.");
        var url = String(body, "polling_url") ?? $"https://api.bfl.ai/v1/get_result?id={Uri.EscapeDataString(taskId)}";
        return (taskId, PollUri(url));
    }

    private static Uri PollUri(string url)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out var target) || !SafeHttps(target)
            || !(target.Host == "api.bfl.ai" || target.Host.StartsWith("api.", StringComparison.Ordinal) && target.Host.EndsWith(".bfl.ai", StringComparison.Ordinal))
            || target.AbsolutePath != "/v1/get_result")
            throw new IOException("Black Forest Labs returned an invalid polling URL.");
        return target;
    }

    private static bool SafeHttps(Uri uri) => uri.Scheme == Uri.UriSchemeHttps && uri.IsDefaultPort && uri.UserInfo.Length == 0 && uri.Fragment.Length == 0;

    private static async Task<(string Status, JsonElement Body)> PollAsync(HttpClient client, Uri target, Action<HttpRequestHeaders> auth, CancellationToken ct)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, target);
        auth(request.Headers);
        using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
        JsonElement body = default;
        try { body = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct); }
        catch (JsonException) when (!response.IsSuccessStatusCode) { }
        var status = String(body, "status");
        // Moderation and task failures can arrive in a structured non-2xx response.
        if (!Failed(status) && !response.IsSuccessStatusCode)
            throw new HttpRequestException("Black Forest Labs polling failed.", null, response.StatusCode);
        return (status ?? throw new IOException("Black Forest Labs did not return a task status."), body);
    }

    private static bool Ready(string? status) => string.Equals(status, "Ready", StringComparison.OrdinalIgnoreCase);
    private static bool Failed(string? status) => status?.ToLowerInvariant() is "error" or "request moderated" or "content moderated" or "task not found" or "failed" or "canceled" or "cancelled";

    private async Task<(string Data, string Mime)> DownloadAsync(Uri target, string fallback, int limit, CancellationToken ct)
    {
        if (!SafeHttps(target) || !target.Host.EndsWith(".bfl.ai", StringComparison.Ordinal)
            || !(target.Host.StartsWith("delivery.", StringComparison.Ordinal) || target.Host.StartsWith("delivery-", StringComparison.Ordinal)))
            throw new IOException("Black Forest Labs returned an invalid asset URL.");
        using var client = clients.CreateClient(AssetClient);
        using var request = new HttpRequestMessage(HttpMethod.Get, target);
        using var response = await NativeHttp.SendAsync(client, request, ct);
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        var mime = response.Content.Headers.ContentType?.MediaType;
        if (mime is null || !mime.StartsWith(fallback[..(fallback.IndexOf('/') + 1)], StringComparison.OrdinalIgnoreCase)) mime = fallback;
        return (await BoundedBase64Async(stream, limit, ct), mime);
    }

    private static (string Data, string Mime) Inline(string value, string fallback, int limit)
    {
        var comma = value.StartsWith("data:", StringComparison.OrdinalIgnoreCase) ? value.IndexOf(',') : -1;
        var mime = fallback;
        if (comma >= 0)
        {
            var header = value[5..comma].Split(';');
            if (header.Length != 2 || !header[1].Equals("base64", StringComparison.OrdinalIgnoreCase)
                || !header[0].StartsWith(fallback[..(fallback.IndexOf('/') + 1)], StringComparison.OrdinalIgnoreCase)
                || !MediaTypeHeaderValue.TryParse(header[0], out _))
                throw new IOException("Black Forest Labs returned an invalid media data URL.");
            mime = header[0];
            value = value[(comma + 1)..];
        }
        if (value.Length > ((long)limit + 2) / 3 * 4 || string.IsNullOrWhiteSpace(value) || !Convert.IsBase64String(value))
            throw new IOException("Black Forest Labs returned invalid or oversized inline media.");
        return (value, mime);
    }

    private static string Reference(string? value)
    {
        if (string.IsNullOrWhiteSpace(value)) throw new ProviderRequestException("A media reference requires inline data or an HTTP(S) URL.");
        var data = value.Trim();
        if (data.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
        {
            var comma = data.IndexOf(',');
            if (comma < 0 || !data[..comma].EndsWith(";base64", StringComparison.OrdinalIgnoreCase) || comma == data.Length - 1)
                throw new ProviderRequestException("Invalid media data URL.");
            return data[(comma + 1)..];
        }
        // References go to BFL as input; the gateway never fetches user-supplied URLs.
        return data;
    }

    private static decimal? Cost(JsonElement body) => Property(body, "cost") is { ValueKind: JsonValueKind.Number } cost
        && cost.TryGetDecimal(out var credits) && credits >= 0 ? credits / 100m : null;

    private static Dictionary<string, JsonElement> Metadata(JsonElement raw, decimal? cost = null)
    {
        var metadata = new Dictionary<string, JsonElement> { [Id] = raw };
        if (cost is { } amount) metadata["gateway"] = JsonSerializer.SerializeToElement(new { cost = amount }, Json);
        return metadata;
    }

    private static JsonElement WithoutMedia(JsonElement body, IEnumerable<string> fields)
    {
        // The existing media payload already contains the result. Don't duplicate inline
        // data or expose signed delivery URLs in the metadata for the same result.
        var copy = JsonObject.Create(body)!;
        if (copy["result"] is JsonObject result)
            foreach (var field in fields) result.Remove(field);
        else if (copy.ContainsKey("result")) copy.Remove("result");
        foreach (var field in fields) copy.Remove(field);
        return JsonSerializer.SerializeToElement(copy, Json);
    }

    private static object Warning(string feature, string? details = null) => new { type = "unsupported", feature, details };
}
