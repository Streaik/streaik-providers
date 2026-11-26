using System.Globalization;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class PicsartProviderRegistration
{
    public static IServiceCollection AddPicsart(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, PicsartProvider.Id, "https://genai-api.picsart.io/v1/");
        return services.AddProvider<PicsartProvider>(PicsartProvider.Id);
    }
}

internal sealed class PicsartProvider(IHttpClientFactory clients, IProviderCredentials credentials, IConfiguration configuration)
    : IModelListingProvider, IImageProvider
{
    internal const string Id = "picsart";
    private const int ImageLimit = 24 * 1024 * 1024;
    private static readonly HashSet<string> Controls = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "prompt", "n", "size", "aspect_ratio", "aspectRatio", "seed", "stream", "partial_images", "response_format",
        "files", "images", "mask", "reference_image", "reference_image_url", "count", "x-picsart-api-key"
    };

    public IAsyncEnumerable<Model> ListModelsAsync(CancellationToken ct) => StaticModelListing.ListAsync(Id, credentials.GetApiKey(Id), ct);

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => GenerateAsync(request.Model, request.Prompt, request.Size, request.N, request.AdditionalProperties, [], [], hasMask: false, ct);

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => GenerateAsync(request.Model, request.Prompt, request.Size, request.N, request.AdditionalProperties, request.ImageFiles?.ToArray() ?? [],
            request.Images ?? [], request.MaskFile is not null || request.Mask is not null, ct);

    private async Task<OpenAIImagesResponse> GenerateAsync(string? model, string? prompt, string? size, int? count, Dictionary<string, JsonElement>? extra,
        IFormFile[] files, OpenAIImageReference[] references, bool hasMask, CancellationToken ct)
    {
        var route = model?.Split('/', 2);
        if (route is not ["text2image" or "text2sticker" or "logo", { Length: > 0 }] || string.IsNullOrWhiteSpace(route[1]))
            throw new ProviderRequestException("Use a Picsart model in the form text2image/<model>, text2sticker/<model>, or logo/<model>.", 422);
        var endpoint = route[0];
        var logo = endpoint == "logo";
        var raw = new Dictionary<string, JsonElement>(NativeMediaOptions.Read(Id, extra, new HashSet<string>(StringComparer.OrdinalIgnoreCase)), StringComparer.OrdinalIgnoreCase);
        var options = ToObject(raw.Where(pair => !Controls.Contains(pair.Key)).ToDictionary());
        options["model"] = route[1];
        if (count is { } n) options["count"] = n;
        var warnings = new List<object>();
        if (hasMask) warnings.Add(new { type = "unsupported", feature = "mask" });
        if (raw.ContainsKey("seed")) warnings.Add(new { type = "unsupported", feature = "seed" });
        var aspect = Value("aspect_ratio") ?? Value("aspectRatio");
        using var content = logo ? LogoPayload() : ImagePayload();
        using var client = clients.CreateClient(Id);
        var key = RequiredKey(credentials, Id);
        var submitted = await SendJsonAsync(HttpMethod.Post, endpoint, content, ct);
        var id = String(submitted, "inference_id") ?? String(submitted, "id");
        if (string.IsNullOrWhiteSpace(id) || id is "." or "..") throw new JsonException("Picsart returned an invalid inference ID.");
        var completed = await PollAsync($"{endpoint}/inferences/{Uri.EscapeDataString(id)}", ct);
        var images = new List<OpenAIImageData>();
        string? format = null;
        var remaining = ImageLimit;
        foreach (var item in Array(Property(completed, "data")))
        {
            if (String(item, "url") is not { Length: > 0 } url) continue;
            var (base64, detected) = await DownloadAsync(client, url, remaining, ct);
            remaining -= (base64.Length / 4) * 3 - (base64.EndsWith("==", StringComparison.Ordinal) ? 2 : base64.EndsWith('=') ? 1 : 0);
            if (format is not null && format != detected) throw new JsonException("Picsart returned mixed image formats.");
            format = detected;
            images.Add(new() { B64Json = base64 });
        }
        if (images.Count == 0) throw new JsonException("Picsart returned no images.");
        var metadata = ToObject(completed);
        if (metadata["data"] is JsonArray data)
            foreach (var image in data.OfType<JsonObject>()) image.Remove("url"); // Media is already returned inline.
        var extensions = new Dictionary<string, JsonElement> { ["providerMetadata"] = JsonSerializer.SerializeToElement(NativeMediaOptions.Metadata(Id, metadata), Json) };
        if (warnings.Count != 0) extensions["warnings"] = JsonSerializer.SerializeToElement(warnings, Json);
        return new() { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Data = images, OutputFormat = format, AdditionalProperties = extensions };

        string? Value(string name) => raw.TryGetValue(name, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;

        HttpContent ImagePayload()
        {
            if (string.IsNullOrWhiteSpace(prompt)) throw new ProviderRequestException("Picsart image generation requires a prompt.");
            if (files.Length + references.Length > 0) warnings.Add(new { type = "unsupported", feature = "files" });
            options["prompt"] = prompt;
            if (size is { Length: > 0 } && !size.Equals("auto", StringComparison.OrdinalIgnoreCase))
            {
                if (!Dimensions(size, ['x', 'X', ':'], out var width, out var height)) throw new ProviderRequestException("Invalid Picsart image size.");
                options["width"] = width;
                options["height"] = height;
            }
            else if (!string.IsNullOrWhiteSpace(aspect))
            {
                if (Dimensions(aspect, [':'], out var a, out var b))
                {
                    var scale = 1024.0 / Math.Max(a, b);
                    options["width"] = Math.Clamp((int)Math.Round(a * scale), 64, 1024);
                    options["height"] = Math.Clamp((int)Math.Round(b * scale), 64, 1024);
                }
                else warnings.Add(new { type = "unsupported", feature = "aspectRatio" });
            }
            return JsonContent.Create(options, options: Json);
        }

        HttpContent LogoPayload()
        {
            var brand = Value("brand_name");
            var description = Value("business_description");
            if (string.IsNullOrWhiteSpace(brand)) brand = prompt;
            if (string.IsNullOrWhiteSpace(description)) description = prompt;
            if (string.IsNullOrWhiteSpace(brand) || string.IsNullOrWhiteSpace(description))
                throw new ProviderRequestException("Picsart logo generation requires a prompt or both brand_name and business_description.");
            options["brand_name"] = brand;
            options["business_description"] = description;
            if (!string.IsNullOrWhiteSpace(size)) warnings.Add(new { type = "unsupported", feature = "size" });
            if (!string.IsNullOrWhiteSpace(aspect)) warnings.Add(new { type = "unsupported", feature = "aspectRatio" });
            var source = Value("reference_image_url");
            if (files.Length + references.Length > 0 && !string.IsNullOrWhiteSpace(source))
                throw new ProviderRequestException("Picsart logos accept one reference image source.");
            if (files.Length + references.Length > 1) warnings.Add(new { type = "unsupported", feature = "files", details = "Only the first reference is used." });
            var form = new MultipartFormDataContent();
            try
            {
                AddFields(form, options);
                if (files.FirstOrDefault() is { } file) AddFile(form, "reference_image", file);
                else if (references.FirstOrDefault() is { } reference)
                {
                    if (reference.FileId is not null) throw new ProviderRequestException("Picsart does not support image file IDs.", 422);
                    if (Uri.TryCreate(reference.ImageUrl, UriKind.Absolute, out var uri) && uri.Scheme is "https" or "http")
                        AddReferenceUrl(form, reference.ImageUrl!);
                    else AddDataImage(form, "reference_image", reference.ImageUrl);
                }
                else if (!string.IsNullOrWhiteSpace(source)) AddReferenceUrl(form, source);
                return form;
            }
            catch { form.Dispose(); throw; }
        }

        async Task<JsonElement> SendJsonAsync(HttpMethod method, string path, HttpContent? body, CancellationToken token)
        {
            using var request = new HttpRequestMessage(method, path) { Content = body };
            request.Headers.Accept.ParseAdd("application/json");
            request.Headers.Add("X-Picsart-API-Key", key);
            using var response = await NativeHttp.SendAsync(client, request, token);
            return await NativeHttp.ReadAsync<JsonElement>(response, Json, token);
        }

        async Task<JsonElement> PollAsync(string path, CancellationToken token)
        {
            using var deadline = CancellationTokenSource.CreateLinkedTokenSource(token);
            deadline.CancelAfter(TimeSpan.FromMinutes(5));
            try
            {
                while (true)
                {
                    var result = await SendJsonAsync(HttpMethod.Get, path, null, deadline.Token);
                    switch (String(result, "status")?.ToLowerInvariant())
                    {
                        case "completed": case "succeeded": case "success": return result;
                        case "failed": case "error": case "cancelled": case "canceled": throw new ProviderRequestException("Picsart inference failed.", 502);
                        case null: throw new JsonException("Picsart inference response has no status.");
                    }
                    await Task.Delay(TimeSpan.FromSeconds(2), deadline.Token);
                }
            }
            catch (OperationCanceledException) when (!token.IsCancellationRequested && deadline.IsCancellationRequested)
            { throw new ProviderRequestException("Picsart inference timed out.", 504); }
        }
    }

    private static bool Dimensions(string size, char[] separators, out int width, out int height)
    {
        width = height = 0;
        var parts = size.Split(separators, StringSplitOptions.TrimEntries);
        return parts.Length == 2 && int.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out width) && width > 0
            && int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out height) && height > 0;
    }

    private static void AddReferenceUrl(MultipartFormDataContent form, string reference)
    {
        if (!Uri.TryCreate(reference, UriKind.Absolute, out var uri) || uri.Scheme is not ("https" or "http") || uri.UserInfo.Length != 0)
            throw new ProviderRequestException("Invalid Picsart reference image URL.");
        form.Add(new StringContent(reference), "reference_image_url"); // Picsart, not the gateway, fetches user-supplied references.
    }

    private async Task<(string Base64, string Format)> DownloadAsync(HttpClient client, string url, int limit, CancellationToken ct)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) || uri.Scheme != Uri.UriSchemeHttps || !uri.IsDefaultPort
            || uri.UserInfo.Length != 0 || uri.Fragment.Length != 0 || uri.HostNameType != UriHostNameType.Dns || !IsAssetHost(uri.IdnHost))
            throw new JsonException("Untrusted Picsart image URL.");
        using var request = new HttpRequestMessage(HttpMethod.Get, uri);
        using var response = await NativeHttp.SendAsync(client, request, ct);
        if (response.Content.Headers.ContentLength > limit) throw new JsonException("Oversized Picsart image download.");
        var format = response.Content.Headers.ContentType?.MediaType?.ToLowerInvariant() switch
        {
            "image/png" => "png", "image/jpeg" or "image/jpg" => "jpeg", "image/webp" => "webp",
            null or "application/octet-stream" => Path.GetExtension(uri.AbsolutePath).ToLowerInvariant() switch { ".jpeg" or ".jpg" => "jpeg", ".webp" => "webp", _ => "png" },
            _ => throw new JsonException("Invalid Picsart image content type.")
        };
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        return (await BoundedBase64Async(stream, limit, ct), format);
    }

    private bool IsAssetHost(string host) => host.EndsWith(".picsart.com", StringComparison.OrdinalIgnoreCase)
        || host.EndsWith(".picsart.io", StringComparison.OrdinalIgnoreCase)
        || configuration.GetSection("Providers:picsart:AssetHosts").GetChildren().Any(entry => string.Equals(entry.Value, host, StringComparison.OrdinalIgnoreCase));
}
