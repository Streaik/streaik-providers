using System.Globalization;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class RecraftProviderRegistration
{
    public static IServiceCollection AddRecraft(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, RecraftProvider.Id, "https://external.api.recraft.ai/");
        ProviderServiceCollectionExtensions.AddClient(services, RecraftProvider.AssetClient, "https://img.recraft.ai/");
        return services.AddProvider<RecraftProvider>(RecraftProvider.Id);
    }
}

internal sealed class RecraftProvider(IHttpClientFactory clients, IProviderCredentials credentials) : IModelListingProvider, IImageProvider
{
    internal const string Id = "recraft";
    internal const string AssetClient = "recraft-assets";
    private const int ImageLimit = 32 * 1024 * 1024;
    private const string ImagePrefix = "v1/images/";
    private static readonly (string Id, string Name)[] Models =
    [
        ("recraftv4_1", "Recraft V4.1"), ("recraftv4_1_pro", "Recraft V4.1 Pro"),
        ("recraftv4", "Recraft V4"), ("recraftv4_pro", "Recraft V4 Pro"),
        ("recraftv3", "Recraft V3"), ("recraftv2", "Recraft V2")
    ];
    private static readonly Dictionary<string, Operation> Operations = new(StringComparer.Ordinal)
    {
        ["vectorize"] = new("Vectorize", "file", Format: "svg"),
        ["generateBackground"] = new("Generate background", "image", Prompt: true, Mask: true),
        ["removeBackground"] = new("Remove background", "file"),
        ["crispUpscale"] = new("Crisp upscale", "file"),
        ["creativeUpscale"] = new("Creative upscale", "file"),
        ["eraseRegion"] = new("Erase region", "image", Mask: true),
        ["variateImage"] = new("Variate image", "image", Size: true)
    };
    private sealed record Operation(string Name, string FileField, bool Prompt = false, bool Mask = false, bool Size = false, string Format = "png");

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        foreach (var (id, name) in Models.Concat(Operations.Select(item => (item.Key, item.Value.Name))))
        {
            ct.ThrowIfCancellationRequested();
            yield return new Model { Id = id, Name = name, Type = "image", OwnedBy = "Recraft" };
        }
        await Task.CompletedTask;
    }

    public async Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        ValidateCount(request.N, request.Stream);
        if (Operations.ContainsKey(request.Model)) throw new ProviderRequestException("This Recraft operation requires an input image.");
        var fields = Fields(request.AdditionalProperties, allowStrength: false);
        fields["model"] = request.Model;
        fields["prompt"] = request.Prompt;
        if (request.Style is { } style) fields["style"] = style;
        if (request.N is { } count) fields["n"] = count;
        if (request.Size is { } size) fields["size"] = size;
        fields["response_format"] = "url";
        using var content = JsonContent.Create(fields, options: Json);
        return await SendAsync("generations", content, request.Size, "png", [], headers, ct);
    }

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        ValidateCount(request.N, request.Stream);
        var key = RequiredKey(credentials, Id);
        if ((request.Images?.Length ?? 0) + (request.ImageFiles?.Length ?? 0) != 1)
            throw new ProviderRequestException("Recraft edits require exactly one input image.");
        if (request.MaskFile is not null && request.Mask is not null) throw new ProviderRequestException("Provide only one mask.");
        var masked = request.MaskFile is not null || request.Mask is not null;
        var operation = Operations.GetValueOrDefault(request.Model);
        if (operation?.Mask == true && !masked) throw new ProviderRequestException($"Recraft '{request.Model}' requires a mask.");

        var fields = Fields(request.AdditionalProperties, allowStrength: true);
        if (operation is null && !masked && !ValidStrength(fields["strength"]))
            throw new ProviderRequestException("Unmasked Recraft edits require a numeric strength between 0 and 1.");
        if (operation is null) fields["model"] = request.Model;
        if (operation?.Prompt != false) fields["prompt"] = request.Prompt;
        if (request.N is { } count) fields["n"] = count;
        if (operation?.Size == true && request.Size is { } size) fields["size"] = size;
        fields["response_format"] = "url";

        List<object> warnings = [];
        if (operation is { Prompt: false } && !string.IsNullOrWhiteSpace(request.Prompt)) warnings.Add(new { type = "unsupported", feature = "prompt" });
        if (operation is { Size: false } && request.Size is not null) warnings.Add(new { type = "unsupported", feature = "size" });
        if (operation is { Mask: false } && masked) warnings.Add(new { type = "unsupported", feature = "mask" });

        using var form = new MultipartFormDataContent();
        AddFields(form, fields);
        await AddImageAsync(form, operation?.FileField ?? "image", request.ImageFiles?.FirstOrDefault(), request.Images?.FirstOrDefault(), ct);
        if (masked && operation?.Mask != false) await AddImageAsync(form, "mask", request.MaskFile, request.Mask, ct);
        var endpoint = operation is not null ? request.Model : masked ? "inpaint" : "imageToImage";
        return await SendAsync(endpoint, form, request.Size, operation?.Format ?? "png", warnings, headers, ct, key);
    }

    private static JsonObject Fields(Dictionary<string, JsonElement>? extensions, bool allowStrength)
    {
        var result = new JsonObject();
        foreach (var (name, value) in extensions ?? [])
        {
            var field = name.ToLowerInvariant();
            if (field is "style" or "style_id" or "negative_prompt" or "random_seed" or "text_layout" or "controls" || field == "strength" && allowStrength)
                result[field] = Copy(value);
        }
        return result;
    }

    // Multipart extension fields arrive as strings, while JSON and Vercel requests can carry a number.
    private static bool ValidStrength(JsonNode? node) => node is JsonValue value &&
        (value.TryGetValue<double>(out var strength) || value.TryGetValue<string>(out var text)
            && double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out strength)) && strength is >= 0 and <= 1;

    private static void ValidateCount(int? count, bool? stream)
    {
        if (count is < 1 or > 6) throw new ProviderRequestException("Recraft image count must be between 1 and 6.");
        if (stream == true && count is > 1) throw new ProviderRequestException("Recraft image batches must be non-streaming.");
    }

    private async Task AddImageAsync(MultipartFormDataContent form, string field, IFormFile? upload, OpenAIImageReference? reference, CancellationToken ct)
    {
        if (upload is not null)
        {
            if (upload.Length <= 0 || upload.Length > ImageLimit) throw new ProviderRequestException("Invalid Recraft image upload size.");
            AddFile(form, field, upload);
        }
        else if (reference is { FileId: null, ImageUrl: { } url })
        {
            if (AssetUri(url) is { } target)
            {
                var (base64, format) = await DownloadAsync(target, "png", ct);
                AddFile(form, field, new MemoryStream(Convert.FromBase64String(base64), writable: false), Mime(format), $"{field}.{format}");
            }
            else
            {
                if (url.Length > ((ImageLimit + 2) / 3) * 4 + 64) throw new ProviderRequestException("Recraft image exceeds the inline size limit.", 413);
                AddDataImage(form, field, url);
            }
        }
        else throw new ProviderRequestException("Recraft requires an uploaded or inline image, or a Recraft asset URL.");
    }

    private async Task<OpenAIImagesResponse> SendAsync(string endpoint, HttpContent content, string? size, string fallback,
        List<object> warnings, IReadOnlyDictionary<string, string> headers, CancellationToken ct, string? key = null)
    {
        using var client = clients.CreateClient(Id);
        var root = await JsonAsync<JsonElement>(client, HttpMethod.Post, ImagePrefix + endpoint, content, "Bearer",
            key ?? RequiredKey(credentials, Id), headers, ct);
        var image = Property(root, "image");
        var data = Property(root, "data");
        var items = data.ValueKind == JsonValueKind.Array ? data.EnumerateArray().ToArray()
            : image.ValueKind is JsonValueKind.Object or JsonValueKind.String ? new[] { image } : [];
        if (items.Length is 0 or > 6) throw new JsonException("Recraft did not return a valid image count.");
        var images = new List<OpenAIImageData>();
        string? outputFormat = null;
        foreach (var item in items)
        {
            var inline = String(item, "b64_json");
            var uri = String(item, "url");
            if (item.ValueKind == JsonValueKind.String)
            {
                var value = item.GetString()!;
                if (Uri.TryCreate(value, UriKind.Absolute, out var candidate) && candidate.Scheme != "data") uri = value;
                else inline = value;
            }
            var (base64, format) = inline is { Length: > 0 } ? Inline(inline, fallback)
                : AssetUri(uri) is { } asset ? await DownloadAsync(asset, fallback, ct)
                : throw new JsonException("Recraft did not return inline image data or a trusted image URL.");
            outputFormat ??= format;
            if (outputFormat != format) throw new JsonException("Recraft returned inconsistent image formats.");
            images.Add(new OpenAIImageData { B64Json = base64, RevisedPrompt = String(item, "revised_prompt") });
        }

        var extensions = new Dictionary<string, JsonElement>
        {
            ["providerMetadata"] = JsonSerializer.SerializeToElement(new Dictionary<string, object?> { [Id] = SafeMetadata(root, response: true) }, Json)
        };
        if (warnings.Count > 0) extensions["warnings"] = JsonSerializer.SerializeToElement(warnings, Json);
        var created = Property(root, "created");
        var usage = Property(root, "usage");
        return new OpenAIImagesResponse { Created = created.ValueKind == JsonValueKind.Number && created.TryGetInt64(out var timestamp)
                ? timestamp : DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Data = images, OutputFormat = outputFormat, Size = size,
            Usage = usage.ValueKind == JsonValueKind.Object ? usage.Deserialize<OpenAIImageUsage>(Json) : null, AdditionalProperties = extensions };
    }

    private static Uri? AssetUri(string? value) => Uri.TryCreate(value, UriKind.Absolute, out var uri)
        && uri.Scheme == Uri.UriSchemeHttps && uri.IsDefaultPort && uri.UserInfo.Length == 0 && uri.Fragment.Length == 0
        && uri.Host is "img.recraft.ai" or "cdn.recraft.ai" ? uri : null;

    private async Task<(string Base64, string Format)> DownloadAsync(Uri target, string fallback, CancellationToken ct)
    {
        // Never copy API credentials or follow redirects on the asset connection.
        using var client = clients.CreateClient(AssetClient);
        using var request = new HttpRequestMessage(HttpMethod.Get, target);
        using var response = await NativeHttp.SendAsync(client, request, ct);
        var mime = response.Content.Headers.ContentType?.MediaType;
        var format = mime is null or "application/octet-stream"
            ? FormatFromExtension(target.AbsolutePath) ?? fallback
            : Format(mime) ?? throw new JsonException("Recraft asset has an unsupported media type.");
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        return (await BoundedBase64Async(stream, ImageLimit, ct), format);
    }

    private static (string Base64, string Format) Inline(string value, string fallback)
    {
        if (value.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
        {
            var comma = value.IndexOf(',');
            if (comma is < 12 or > 64 || !value[..comma].EndsWith(";base64", StringComparison.OrdinalIgnoreCase))
                throw new JsonException("Invalid Recraft data URL.");
            fallback = Format(value[5..(comma - 7)]) ?? throw new JsonException("Unsupported Recraft image type.");
            value = value[(comma + 1)..];
        }
        if (value.Length > ((ImageLimit + 2) / 3) * 4) throw new JsonException("Recraft image exceeds the inline size limit.");
        try
        {
            var bytes = Convert.FromBase64String(value);
            if (bytes.Length is 0 or > ImageLimit) throw new JsonException("Invalid Recraft image size.");
            return (Convert.ToBase64String(bytes), fallback);
        }
        catch (FormatException) { throw new JsonException("Invalid base64 image from Recraft."); }
    }

    private static string? Format(string mime) => mime.ToLowerInvariant() switch
    {
        "image/png" => "png", "image/jpeg" => "jpeg", "image/webp" => "webp", "image/svg+xml" => "svg", _ => null
    };
    private static string? FormatFromExtension(string path) => Path.GetExtension(path).ToLowerInvariant() switch
    {
        ".png" => "png", ".jpeg" or ".jpg" => "jpeg", ".webp" => "webp", ".svg" => "svg", _ => null
    };
    private static string Mime(string format) => format == "svg" ? "image/svg+xml" : $"image/{format}";

    private static JsonNode? SafeMetadata(JsonElement value, bool response = false) => value.ValueKind switch
    {
        JsonValueKind.Object => new JsonObject(value.EnumerateObject()
            .Where(field => !SensitiveField(field.Name) && !(field.Name == "image" && field.Value.ValueKind == JsonValueKind.String))
            .Select(field => KeyValuePair.Create(field.Name, response && field.Name == "data" && field.Value.ValueKind == JsonValueKind.Array
                ? new JsonArray(field.Value.EnumerateArray().Select(item => item.ValueKind == JsonValueKind.String ? null : SafeMetadata(item)).ToArray())
                : SafeMetadata(field.Value)))),
        JsonValueKind.Array => new JsonArray(value.EnumerateArray().Select(item => SafeMetadata(item)).ToArray()),
        JsonValueKind.String when value.GetString() is { } text && (text.StartsWith("data:", StringComparison.OrdinalIgnoreCase)
            || Uri.TryCreate(text, UriKind.Absolute, out var uri) && uri.Scheme is "http" or "https" && (uri.Query.Length > 0 || uri.UserInfo.Length > 0)) => null,
        _ => Copy(value)
    };

    private static bool SensitiveField(string name) => name.Replace("_", "").Replace("-", "").ToLowerInvariant() is
        "url" or "imageurl" or "b64json" or "authorization" or "apikey" or "token" or "accesstoken" or "cookie" or "setcookie";
}
