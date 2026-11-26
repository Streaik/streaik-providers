using System.Globalization;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class AstriaRegistration
{
    public static IServiceCollection AddAstria(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, "astria", "https://api.astria.ai/");
        services.AddHttpClient("astria-delivery").ConfigurePrimaryHttpMessageHandler(() => new SocketsHttpHandler
            { UseCookies = false, AllowAutoRedirect = false });
        return services.AddProvider<AstriaProvider>("astria");
    }
}

internal sealed partial class AstriaProvider(IHttpClientFactory clients, IProviderCredentials credentials, IConfiguration config)
    : IModelListingProvider, IImageModelRoutingProvider, IVideoProvider
{
    private const string Id = "astria";
    private const int MediaLimit = 25 * 1024 * 1024;
    private Task<JsonElement>? catalog;
    private static readonly HashSet<string> Controls = new(StringComparer.OrdinalIgnoreCase)
    { Id, "model", "prompt", "stream", "partial_images", "files", "images", "mask", "response_format", "size", "n",
        "provider_options", "providerOptions", "authorization", "api-key", "api_key", "apiKey", "x-api-key", "headers" };

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        var root = await CatalogAsync(ct);
        var ids = new HashSet<string>(StringComparer.Ordinal);
        if (Property(root, "models") is { ValueKind: JsonValueKind.Object } images)
            foreach (var entry in images.EnumerateObject())
                if (TuneId(entry.Value) is { } id && ids.Add(id))
                    yield return new() { Id = id, Name = entry.Name, Type = "image", OwnedBy = Id };
        if (Property(root, "video_model_families") is { ValueKind: JsonValueKind.Object } videos)
            foreach (var entry in videos.EnumerateObject())
                if (entry.Value.ValueKind == JsonValueKind.Object && !string.IsNullOrWhiteSpace(entry.Name) && ids.Add(entry.Name))
                    yield return new() { Id = entry.Name, Name = String(entry.Value, "title") ?? entry.Name, Type = "video", OwnedBy = Id };
    }

    public async Task<bool> IsImageModelAsync(string model, CancellationToken ct)
        => PositiveId(model) is not null || TuneId(Property(Property(await CatalogAsync(ct), "models"), model)) is not null;

    private async Task<string> ResolveTuneAsync(string model, CancellationToken ct)
        => PositiveId(model) ?? TuneId(Property(Property(await CatalogAsync(ct), "models"), model))
            ?? throw new ProviderRequestException("Unknown Astria image model.");

    private Task<JsonElement> CatalogAsync(CancellationToken ct) => catalog ??= ReadCatalogAsync(ct);

    private async Task<JsonElement> ReadCatalogAsync(CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        return await NativeHttp.GetAsync<JsonElement>(client, "cli/models", _ => { }, ct);
    }

    private static string? TuneId(JsonElement entry) => PositiveId(Property(entry, "tune_id").ToString());
    private static string? PositiveId(string? value)
        => long.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out var number) && number > 0
            ? number.ToString(CultureInfo.InvariantCulture) : null;

    private async Task<JsonElement> SubmitAsync(string tune, MultipartFormDataContent form, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var result = await PromptAsync(HttpMethod.Post, $"tunes/{tune}/prompts", form, headers, ct);
        if (result.ValueKind == JsonValueKind.Array && result.GetArrayLength() == 1) result = result[0];
        _ = PromptId(result);
        return result;
    }

    private Task<JsonElement> PollAsync(string tune, string prompt, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PromptAsync(HttpMethod.Get, $"tunes/{tune}/prompts/{prompt}", null, headers, ct);

    private async Task<JsonElement> PromptAsync(HttpMethod method, string path, HttpContent? body, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        return await JsonAsync<JsonElement>(client, method, path, body, "Bearer", RequiredKey(credentials, Id), headers, ct);
    }

    private static string PromptId(JsonElement root) => PositiveId(Property(root, "id").ToString())
        ?? throw new JsonException("Astria did not return a valid prompt ID.");

    private static IEnumerable<string> Outputs(JsonElement root) => Array(Property(root, "images"))
        .Where(item => item.ValueKind == JsonValueKind.String).Select(item => item.GetString()!).Where(url => !string.IsNullOrWhiteSpace(url));

    private static Dictionary<string, JsonElement> Metadata(JsonElement root)
    {
        var options = root.EnumerateObject().ToDictionary(field => field.Name, field => field.Value);
        if (Property(root, "cost_mc") is { ValueKind: JsonValueKind.Number } cost && cost.TryGetDecimal(out var value))
            options.TryAdd("cost", JsonSerializer.SerializeToElement(value / 100000m));
        return new() { [Id] = JsonSerializer.SerializeToElement(options, Json) };
    }

    private static Dictionary<string, JsonElement> NativeOptions(Dictionary<string, JsonElement>? fields, bool scopedOnly = false)
    {
        var options = new Dictionary<string, JsonElement>(StringComparer.OrdinalIgnoreCase);
        if (fields is null) return options;
        foreach (var wrapper in new[] { "provider_options", "providerOptions" })
            if (fields.TryGetValue(wrapper, out var entry) && Property(entry, Id) is { ValueKind: not JsonValueKind.Undefined } selected)
                AddScope(selected);
        if (fields.TryGetValue(Id, out var scope)) AddScope(scope);
        if (!scopedOnly)
            foreach (var (name, value) in fields) Add(name, value);
        return options;

        void AddScope(JsonElement value)
        {
            if (value.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Astria options must be an object.");
            foreach (var field in value.EnumerateObject()) Add(field.Name, field.Value);
        }
        void Add(string name, JsonElement value)
        {
            if (!Controls.Contains(name)) options[name] = value;
        }
    }

    private static void Set<T>(Dictionary<string, JsonElement> options, string name, T value)
    {
        if (value is not null) options[name] = JsonSerializer.SerializeToElement(value, Json);
    }

    private static void AddOptions(MultipartFormDataContent form, Dictionary<string, JsonElement> options)
    {
        foreach (var (name, value) in options) AddField(form, $"prompt[{FieldName(name)}]", value);
    }

    private static string FieldName(string name)
    {
        if (name.Length == 0 || name.Any(c => char.IsControl(c) || c is '[' or ']' or '"' or '\\'))
            throw new ProviderRequestException("Invalid Astria multipart option name.");
        return name;
    }

    private static void AddField(MultipartFormDataContent form, string key, JsonElement value)
    {
        switch (value.ValueKind)
        {
            case JsonValueKind.Object:
                foreach (var entry in value.EnumerateObject()) AddField(form, $"{key}[{FieldName(entry.Name)}]", entry.Value);
                break;
            case JsonValueKind.Array:
                foreach (var entry in value.EnumerateArray()) AddField(form, key + "[]", entry);
                break;
            case JsonValueKind.Undefined: break;
            default:
                form.Add(new StringContent(value.ValueKind switch { JsonValueKind.String => value.GetString()!, JsonValueKind.Null => "", _ => value.GetRawText() }), key);
                break;
        }
    }

    private static bool IsRemote(string? value) => Uri.TryCreate(value, UriKind.Absolute, out var url) &&
        url.Scheme is "http" or "https" && url.UserInfo.Length == 0;

    private static void RemoveMedia(Dictionary<string, JsonElement> options, string field)
    {
        options.Remove(field);
        options.Remove(field + "_url");
        options.Remove(field + "_base64");
    }

    private static void AddMedia(MultipartFormDataContent form, string field, string? data, string? mime)
    {
        if (IsRemote(data)) form.Add(new StringContent(data!), $"prompt[{field}_url]");
        else AddBinary(form, $"prompt[{field}]", data, mime);
    }

    private static void AddBinary(MultipartFormDataContent form, string field, string? data, string? mime)
    {
        if (string.IsNullOrWhiteSpace(data)) throw new ProviderRequestException("Missing Astria media input.");
        if (data.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
        {
            var comma = data.IndexOf(',');
            if (comma is < 12 or > 128 || !data[..comma].EndsWith(";base64", StringComparison.OrdinalIgnoreCase))
                throw new ProviderRequestException("Invalid Astria media data URL.");
            mime ??= data[5..(comma - 7)];
            data = data[(comma + 1)..];
        }
        if (data.Length > MediaLimit * 4 / 3 + 8) throw new ProviderRequestException("Astria media input is too large.", 413);
        byte[] bytes;
        try { bytes = Convert.FromBase64String(data); }
        catch (FormatException) { throw new ProviderRequestException("Invalid Astria media base64."); }
        if (bytes.Length > MediaLimit) throw new ProviderRequestException("Astria media input is too large.", 413);
        if (bytes.Length == 0) throw new ProviderRequestException("Empty Astria media input.");
        AddFile(form, field, new MemoryStream(bytes, writable: false), mime ?? "image/png", "input");
    }

    private async Task<(string Data, string Mime, int Length)?> DownloadAsync(string url, bool video, int limit, CancellationToken ct)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out var target) || target.Scheme != Uri.UriSchemeHttps || !target.IsDefaultPort
            || target.UserInfo.Length != 0 || target.Fragment.Length != 0 || target.HostNameType != UriHostNameType.Dns
            || !(target.IdnHost == "astria.ai" || target.IdnHost.EndsWith(".astria.ai", StringComparison.Ordinal)
                || (config["Providers:astria:DeliveryHosts"] ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Contains(target.IdnHost, StringComparer.OrdinalIgnoreCase)))
            throw new IOException("Untrusted Astria delivery host.");
        using var client = clients.CreateClient("astria-delivery");
        using var request = new HttpRequestMessage(HttpMethod.Get, target);
        using var response = await NativeHttp.SendAsync(client, request, ct);
        var mime = response.Content.Headers.ContentType?.MediaType?.ToLowerInvariant();
        if (mime is null or "application/octet-stream") mime = Path.GetExtension(target.AbsolutePath).ToLowerInvariant() switch
        {
            ".mp4" => "video/mp4", ".webm" => "video/webm", ".mov" => "video/quicktime", ".jpg" or ".jpeg" => "image/jpeg",
            ".png" => "image/png", ".webp" => "image/webp", ".gif" => "image/gif", ".avif" => "image/avif", _ => video ? null : "image/png"
        };
        // Astria can return the intermediate image while the corresponding video is still rendering.
        if (video && (mime is null || mime.StartsWith("image/", StringComparison.Ordinal))) return null;
        if (mime?.StartsWith(video ? "video/" : "image/", StringComparison.Ordinal) != true)
            throw new IOException("Unexpected Astria media type.");
        if (limit <= 0 || response.Content.Headers.ContentLength > limit) throw new IOException("Astria output is too large.");
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        var data = await BoundedBase64Async(stream, limit, ct);
        var length = data.Length / 4 * 3 - (data.EndsWith("==", StringComparison.Ordinal) ? 2 : data.EndsWith('=') ? 1 : 0);
        return (data, mime, length);
    }
}
