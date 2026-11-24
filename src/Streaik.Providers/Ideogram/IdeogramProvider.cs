using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class IdeogramProviderRegistration
{
    public static IServiceCollection AddIdeogram(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, IdeogramProvider.Id, "https://api.ideogram.ai/");
        ProviderServiceCollectionExtensions.AddClient(services, IdeogramProvider.AssetClient, "https://ideogram.ai/");
        return services.AddProvider<IdeogramProvider>(IdeogramProvider.Id);
    }
}

internal sealed class IdeogramProvider(IHttpClientFactory clients, IProviderCredentials credentials) : IModelListingProvider, IImageProvider
{
    internal const string Id = "ideogram";
    internal const string AssetClient = "ideogram-assets";
    private const int MediaLimit = 25 * 1024 * 1024;
    private const string V3 = "ideogram-v3", V4 = "ideogram-v4";
    private const string Transparent = V3 + "/generate-transparent", Reframe = V3 + "/reframe", Replace = V3 + "/replace-background", Upscale = "upscale";
    private static readonly (string Id, string Name)[] Models =
    [
        (V4, "Ideogram 4.0"), (V3, "Ideogram 3.0"), (Transparent, "Ideogram 3.0 Generate (Transparent Background)"),
        (Reframe, "Ideogram 3.0 Reframe"), (Replace, "Ideogram 3.0 Replace Background"), (Upscale, "Upscale")
    ];
    private static readonly string[] ReferenceFields = ["style_reference_images", "character_reference_images", "character_reference_images_mask"];

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        foreach (var (id, name) in Models)
        {
            ct.ThrowIfCancellationRequested();
            yield return new Model { Id = id, Name = name, Type = "image", OwnedBy = "Ideogram" };
        }
        await Task.CompletedTask;
    }

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => RunAsync(request.Model, request.Prompt, request.Size, request.N, request.Stream, request.AdditionalProperties,
            [], [], false, request.Style, headers, ct);

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if ((request.ImageFiles?.Length ?? 0) + (request.Images?.Length ?? 0) == 0)
            throw new ProviderRequestException("Ideogram edits require an input image.");
        return RunAsync(request.Model, request.Prompt, request.Size, request.N, request.Stream, request.AdditionalProperties,
            request.ImageFiles ?? [], request.Images ?? [], request.Mask is not null || request.MaskFile is not null, null, headers, ct);
    }

    private async Task<OpenAIImagesResponse> RunAsync(string model, string? prompt, string? size, int? n, bool? stream,
        Dictionary<string, JsonElement>? extensions, IFormFile[] uploads, OpenAIImageReference[] references, bool mask, string? style,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var count = uploads.Length + references.Length;
        var endpoint = (model, count > 0, mask) switch
        {
            (V3, false, false) => "v1/ideogram-v3/generate",
            (V3, true, false) => "v1/ideogram-v3/remix",
            (V3, true, true) => "v1/edit",
            (V4, false, false) => "v1/ideogram-v4/generate",
            (V4, true, false) => "v1/ideogram-v4/remix",
            (Transparent, false, false) or (Reframe or Replace, true, _) => "v1/" + model,
            (Upscale, true, _) => Upscale,
            (V4, true, true) => throw new ProviderRequestException("Ideogram 4.0 does not provide a masked edit endpoint; use Ideogram 3.0."),
            _ => throw new ProviderRequestException("The Ideogram model does not support the supplied inputs.")
        };
        var key = RequiredKey(credentials, Id);
        var fields = Options(extensions);
        var warnings = new List<object>();
        void Ignore(string feature, string details) => warnings.Add(new { type = "unsupported", feature, details });
        void Drop(string field, string details)
        {
            if (fields.Remove(field)) Ignore(field, details);
        }

        if (n is { } requestedCount && fields["num_images"] is null) fields["num_images"] = requestedCount;
        if (style is not null && fields["style_type"] is null) fields["style_type"] = style;
        if (size is not (null or "auto" or "") && fields["resolution"] is null) fields["resolution"] = size;
        if (fields["aspect_ratio"] is JsonValue ratio && ratio.TryGetValue<string>(out var ratioText))
            fields["aspect_ratio"] = ratioText.Replace(':', 'x');

        var v3Edit = endpoint == "v1/edit";
        if (mask) Ignore("mask", "The mask is not sent to the selected Ideogram endpoint.");
        if (!v3Edit && count > 1) Ignore("files", "This Ideogram operation accepts one input image; used the first.");
        if (model == V4)
        {
            Drop("seed", "Ideogram 4.0 does not accept a seed.");
            Drop("num_images", "Ideogram 4.0 returns one image.");
            Drop("aspect_ratio", "Use a supported resolution for Ideogram 4.0.");
            if (count == 0) Drop("image_weight", "Image weight applies only to remix.");
        }
        if (model == Transparent) Drop("resolution", "Use an aspect ratio for transparent generation.");
        if (model == Reframe && (fields["resolution"] is not JsonValue resolution || !resolution.TryGetValue<string>(out var dimensions)
            || string.IsNullOrWhiteSpace(dimensions))) throw new ProviderRequestException("Ideogram reframe requires a resolution.");
        if (model != V4 || count > 0) Drop("json_prompt", "Structured prompts apply only to Ideogram 4.0 generation.");

        if (model == Upscale)
        {
            // The legacy upscale route takes options as JSON in a multipart field.
            if (fields.Remove("upscale", out var options) && options is not JsonObject)
                throw new ProviderRequestException("Ideogram upscale options must be an object.");
            var upscale = (JsonObject?)options ?? new JsonObject();
            if (!string.IsNullOrWhiteSpace(prompt) && upscale["prompt"] is null) upscale["prompt"] = prompt;
            foreach (var name in new[] { "resemblance", "detail", "magic_prompt_option", "num_images", "seed" })
                if (fields.Remove(name, out var value) && upscale[name] is null) upscale[name] = value;
            CheckCount(upscale["num_images"], stream);
            fields.Remove("resolution");
            fields["image_request"] = upscale;
        }
        else
        {
            CheckCount(fields["num_images"], stream);
            if (model == Reframe)
            {
                if (!string.IsNullOrWhiteSpace(prompt)) Ignore("prompt", "Reframe only uses an image and resolution.");
            }
            else if (fields["json_prompt"] is not null)
            {
                if (!string.IsNullOrWhiteSpace(prompt)) Ignore("prompt", "Structured and text prompts are mutually exclusive; used the structured prompt.");
            }
            else
            {
                if (string.IsNullOrWhiteSpace(prompt)) throw new ProviderRequestException("Ideogram requires a prompt.");
                fields[model == V4 ? "text_prompt" : "prompt"] = prompt;
            }
        }

        using var form = new MultipartFormDataContent();
        foreach (var field in ReferenceFields)
        {
            if (!fields.Remove(field, out var value)) continue;
            if (model is V4 or Transparent or Upscale || v3Edit) { Ignore(field, "The selected Ideogram operation does not accept reference images."); continue; }
            if (value is JsonArray array)
                foreach (var reference in array) AddReference(form, field, reference);
            else AddReference(form, field, value);
        }
        if (fields.Remove("style_codes", out var codes) && codes is JsonArray list)
        {
            foreach (var code in list)
                if (code is JsonValue value && value.TryGetValue<string>(out var text)) form.Add(new StringContent(text), "style_codes");
        }
        else if (codes is not null) fields["style_codes"] = codes;
        AddFields(form, fields);

        if (count > 0)
        {
            var field = model == Upscale ? "image_file" : v3Edit ? "images" : "image";
            foreach (var upload in uploads.Take(v3Edit ? uploads.Length : 1))
            {
                if (upload.Length is <= 0 or > MediaLimit) throw new ProviderRequestException("Invalid Ideogram image upload size.");
                AddFile(form, field, upload);
            }
            foreach (var reference in references.Take(v3Edit ? references.Length : uploads.Length == 0 ? 1 : 0))
            {
                if (reference.FileId is not null) throw new ProviderRequestException("Ideogram requires uploaded or inline image data, not a file ID.");
                AddInline(form, field, reference.ImageUrl);
            }
        }

        using var client = clients.CreateClient(Id);
        using var message = new HttpRequestMessage(HttpMethod.Post, endpoint) { Content = form };
        message.Headers.Add("Api-Key", key);
        message.Headers.Accept.ParseAdd("application/json");
        NativeHttp.ApplyHeaders(message, headers);
        using var response = await NativeHttp.SendAsync(client, message, ct);
        var root = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        return await ReadImagesAsync(root, size, stream == true, warnings, ct);
    }

    private static JsonObject Options(Dictionary<string, JsonElement>? extensions)
    {
        var fields = new JsonObject();
        if (extensions is null) return fields;
        foreach (var name in new[] { "provider_options", "providerOptions", Id })
        {
            if (!extensions.TryGetValue(name, out var value)) continue;
            var scope = name == Id ? value : Property(value, Id);
            if (scope.ValueKind == JsonValueKind.Object)
                foreach (var option in scope.EnumerateObject()) Add(option.Name, option.Value);
        }
        foreach (var (name, value) in extensions) Add(name, value);
        return fields;

        void Add(string name, JsonElement value)
        {
            if (value.ValueKind == JsonValueKind.Null || name.Length == 0 || !name.All(c => char.IsAsciiLetterLower(c) || char.IsAsciiDigit(c) || c == '_')
                || Sensitive(name) || name is Id or "provider_options" or "model" or "prompt" or "text_prompt" or "action" or "image" or "images" or "image_file"
                    or "image_request" or "mask" or "stream" or "partial_images" or "response_format" or "size" or "n") return;
            if (name is "json_prompt" or "upscale" or "style_codes" && value.ValueKind == JsonValueKind.String
                && value.GetString()?.TrimStart() is { Length: > 0 } text && text[0] is '{' or '[')
            {
                try { fields[name] = JsonNode.Parse(text); }
                catch (JsonException) { throw new ProviderRequestException($"Invalid Ideogram {name} JSON."); }
            }
            else fields[name] = Copy(value);
        }
    }

    private static void CheckCount(JsonNode? value, bool? stream)
    {
        if (value is null) return;
        if (value is not JsonValue scalar || !int.TryParse(scalar.ToString(), out var count) || count is < 1 or > 8)
            throw new ProviderRequestException("Ideogram image count must be between 1 and 8.");
        if (stream == true && count > 1) throw new ProviderRequestException("Ideogram image batches must be non-streaming.");
    }

    private static void AddReference(MultipartFormDataContent form, string field, JsonNode? value)
    {
        if (value is not JsonValue item || !item.TryGetValue<string>(out var image))
            throw new ProviderRequestException("Ideogram reference images must be base64 strings.");
        AddInline(form, field, image.StartsWith("data:", StringComparison.Ordinal) ? image : "data:image/png;base64," + image);
    }

    private static void AddInline(MultipartFormDataContent form, string field, string? value)
    {
        if (value?.Length > ((MediaLimit + 2) / 3) * 4 + 64) throw new ProviderRequestException("Ideogram image exceeds the inline size limit.", 413);
        AddDataImage(form, field, value);
    }

    private async Task<OpenAIImagesResponse> ReadImagesAsync(JsonElement body, string? size, bool stream, List<object> warnings, CancellationToken ct)
    {
        var data = Property(body, "data");
        if (data.ValueKind != JsonValueKind.Array || data.GetArrayLength() is < 1 or > 8 || stream && data.GetArrayLength() != 1)
            throw new JsonException("Ideogram returned an invalid image count.");
        using var assetClient = clients.CreateClient(AssetClient);
        var images = new List<OpenAIImageData>();
        string? format = null;
        foreach (var item in data.EnumerateArray())
        {
            if (!Uri.TryCreate(String(item, "url"), UriKind.Absolute, out var asset) || asset.Scheme != Uri.UriSchemeHttps
                || !asset.IsDefaultPort || asset.UserInfo.Length != 0 || asset.Fragment.Length != 0
                || asset.Host is not ("ideogram.ai" or "www.ideogram.ai" or "cdn.ideogram.ai" or "images.ideogram.ai"))
                throw new JsonException("Ideogram did not return a trusted image URL.");
            // Only the provider request carries credentials. The asset client does not follow redirects.
            using var download = new HttpRequestMessage(HttpMethod.Get, asset);
            using var response = await NativeHttp.SendAsync(assetClient, download, ct);
            if (response.Content.Headers.ContentLength > MediaLimit) throw new JsonException("Ideogram image exceeds the inline size limit.");
            var mime = response.Content.Headers.ContentType?.MediaType;
            var actual = mime switch
            {
                "image/png" => "png", "image/jpeg" => "jpeg", "image/webp" => "webp", "image/gif" => "gif",
                null or "application/octet-stream" => Path.GetExtension(asset.AbsolutePath).ToLowerInvariant() switch
                { ".jpg" or ".jpeg" => "jpeg", ".webp" => "webp", ".gif" => "gif", _ => "png" },
                _ => throw new JsonException("Ideogram returned an unsupported image type.")
            };
            format ??= actual;
            if (format != actual) throw new JsonException("Ideogram returned inconsistent image formats.");
            await using var source = await response.Content.ReadAsStreamAsync(ct);
            images.Add(new() { B64Json = await BoundedBase64Async(source, MediaLimit, ct), RevisedPrompt = String(item, "prompt") });
        }

        var extensions = new Dictionary<string, JsonElement>
        {
            ["providerMetadata"] = JsonSerializer.SerializeToElement(new JsonObject { [Id] = SafeMetadata(body) }, Json)
        };
        if (warnings.Count > 0) extensions["warnings"] = JsonSerializer.SerializeToElement(warnings, Json);
        var created = Property(body, "created");
        return new() { Created = created.ValueKind == JsonValueKind.Number && created.TryGetInt64(out var timestamp)
                ? timestamp : DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Data = images, OutputFormat = format, Size = size, AdditionalProperties = extensions };
    }

    private static JsonNode? SafeMetadata(JsonElement value) => value.ValueKind switch
    {
        JsonValueKind.Object => new JsonObject(value.EnumerateObject().Where(field => !Sensitive(field.Name))
            .Select(field => KeyValuePair.Create(field.Name, SafeMetadata(field.Value)))),
        JsonValueKind.Array => new JsonArray(value.EnumerateArray().Select(SafeMetadata).ToArray()),
        JsonValueKind.String when value.GetString() is { } text && (text.StartsWith("data:", StringComparison.OrdinalIgnoreCase)
            || Uri.TryCreate(text, UriKind.Absolute, out var uri) && uri.Scheme is "https" or "http" && (uri.Query.Length > 0 || uri.UserInfo.Length > 0)) => null,
        _ => Copy(value)
    };

    private static bool Sensitive(string name) => name.Replace("_", "").Replace("-", "").ToLowerInvariant() is
        "url" or "imageurl" or "b64json" or "authorization" or "apikey" or "token" or "accesstoken" or "cookie" or "setcookie";
}
