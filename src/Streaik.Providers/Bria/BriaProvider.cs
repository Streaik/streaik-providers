using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class BriaProviderRegistration
{
    public static IServiceCollection AddBria(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, BriaProvider.Id, BriaProvider.BaseAddress);
        services.AddHttpClient(BriaProvider.DeliveryClient)
            .ConfigurePrimaryHttpMessageHandler(() => new SocketsHttpHandler { UseCookies = false, AllowAutoRedirect = false });
        return services.AddProvider<BriaProvider>(BriaProvider.Id);
    }
}

internal sealed partial class BriaProvider(IHttpClientFactory clients, IProviderCredentials credentials, IConfiguration configuration)
    : IModelListingProvider, IImageModelRoutingProvider, IVideoProvider
{
    internal const string Id = "bria";
    internal const string BaseAddress = "https://engine.prod.bria-api.com/v2/";
    internal const string DeliveryClient = "bria-delivery";
    private static readonly Uri Api = new(BaseAddress);
    private const int ImageLimit = 32 * 1024 * 1024;
    private const int VideoLimit = 64 * 1024 * 1024;

    // Native model IDs match the legacy catalog; only these IDs can be used as endpoint paths.
    private static readonly Dictionary<string, ImageWorkflow> Workflows = new(StringComparer.Ordinal)
    {
        ["generate"] = new("Generate", "generateImage", "prompt", Multiple: true, PromptRequired: true),
        ["generate/lite"] = new("Generate Lite", "generateImage", "prompt", Multiple: true, PromptRequired: true),
        ["edit"] = new("Edit", "imageEdit", "instruction", Multiple: true, PromptRequired: true),
        ["edit/add_object_by_text"] = new("Add Object by Text", Prompt: "instruction", Sync: false, PromptRequired: true),
        ["edit/replace_object_by_text"] = new("Replace Object by Text", Prompt: "instruction", Sync: false, PromptRequired: true),
        ["edit/erase_by_text"] = new("Erase by Text", Prompt: "object_name", Sync: false, PromptRequired: true),
        ["edit/blend"] = new("Blend", Prompt: "instruction", Sync: false, PromptRequired: true),
        ["edit/reseason"] = new("Reseason", "reseasonImage", "season", Sync: false, PromptRequired: true),
        ["edit/replace_text"] = new("Replace Text", Prompt: "new_text", Sync: false, PromptRequired: true),
        ["edit/sketch_to_image"] = new("Sketch to Image", Sync: false),
        ["edit/restore"] = new("Restore"),
        ["edit/colorize"] = new("Colorize", "colorize", "style", PromptRequired: true),
        ["edit/restyle"] = new("Restyle", "restyleImage", "style", PromptRequired: true),
        ["edit/relight"] = new("Relight", "relightImage", "light_type", PromptRequired: true),
        ["edit/erase"] = new("Erase", "eraser", MaskRequired: true),
        ["edit/gen_fill"] = new("Generative Fill", "generativeFill", "prompt", MaskRequired: true, PromptRequired: true),
        ["edit/remove_background"] = new("Remove Background", "removeBackground"),
        ["edit/replace_background"] = new("Replace Background", "replaceBackground", "prompt"),
        ["edit/erase_foreground"] = new("Erase Foreground", "eraseForeground"),
        ["edit/blur_background"] = new("Blur Background", "blurBackground"),
        ["edit/expand"] = new("Expand", "expandImage", "prompt"),
        ["edit/enhance"] = new("Enhance", "enhanceImage"),
        ["edit/increase_resolution"] = new("Increase Resolution", "increaseResolution"),
        ["edit/crop_foreground"] = new("Crop Foreground", "cropoutForeground")
    };

    private sealed record ImageWorkflow(string Name, string? OptionGroup = null, string? Prompt = null,
        bool Multiple = false, bool Sync = true, bool PromptRequired = false, bool MaskRequired = false);

    private Action<HttpRequestHeaders> Auth()
    {
        var key = RequiredKey(credentials, Id);
        return h => h.Add("api_token", key);
    }

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        foreach (var (model, workflow) in Workflows)
        {
            ct.ThrowIfCancellationRequested();
            yield return new Model { Id = model, Name = "Bria " + workflow.Name, OwnedBy = Id, Type = "image" };
        }
        foreach (var (model, name) in VideoModels)
        {
            ct.ThrowIfCancellationRequested();
            yield return new Model { Id = model, Name = "Bria " + name, OwnedBy = Id, Type = "video" };
        }
    }

    public Task<bool> IsImageModelAsync(string model, CancellationToken ct) => Task.FromResult(Workflows.ContainsKey(model));

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => ImageAsync(request.Model, request.Prompt, request.N, request.Size,
            request.AdditionalProperties, [], null, headers, ct);

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        _ = Workflow(request.Model);
        var images = new List<string>();
        foreach (var image in request.Images ?? []) images.Add(ImageValue(image));
        foreach (var file in request.ImageFiles ?? []) images.Add(await InlineAsync(file, ct));
        if (request.Mask is not null && request.MaskFile is not null) throw new ProviderRequestException("Supply only one Bria image mask.");
        var mask = request.MaskFile is { } upload ? await InlineAsync(upload, ct) : request.Mask is { } reference ? ImageValue(reference) : null;
        return await ImageAsync(request.Model, request.Prompt, request.N, request.Size,
            request.AdditionalProperties, images, mask, headers, ct);
    }

    private static ImageWorkflow Workflow(string model) => Workflows.TryGetValue(model, out var workflow)
        ? workflow : throw new ProviderRequestException("Unknown Bria image model.");

    private async Task<OpenAIImagesResponse> ImageAsync(string model, string? prompt, int? n, string? size,
        Dictionary<string, JsonElement>? options, IReadOnlyList<string> images, string? mask, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var workflow = Workflow(model);
        if (n is not (null or 1)) throw new ProviderRequestException("Bria returns one image per request.");
        if (model.StartsWith("edit", StringComparison.Ordinal) && images.Count == 0 || !workflow.Multiple && images.Count != 1)
            throw new ProviderRequestException("Bria image editing requires an input image (one for specialty tools).");
        if (workflow.MaskRequired && mask is null) throw new ProviderRequestException("This Bria image workflow requires a mask.");
        var payload = ImageOptions(options, workflow.OptionGroup);
        if (workflow.Prompt is { } field)
        {
            var structuredEdit = model == "edit" && HasText(payload, "structured_instruction");
            // Style selectors may come from provider options; never override the request's actual prompt.
            if (!structuredEdit && !string.IsNullOrWhiteSpace(prompt)
                && (field is not ("season" or "style" or "light_type") || !HasText(payload, field))) payload[field] = prompt;
            if (structuredEdit) payload.Remove(field);
            if (workflow.PromptRequired && !structuredEdit && !HasText(payload, field))
                throw new ProviderRequestException($"This Bria image workflow requires '{field}'.");
        }
        if (images.Count > 0)
        {
            if (workflow.Multiple) payload["images"] = new JsonArray(images.Select(image => (JsonNode?)JsonValue.Create(image)).ToArray());
            else payload["image"] = images[0];
        }
        if (mask is not null && (model == "edit" || workflow.MaskRequired)) payload["mask"] = mask;
        if (model is "generate" or "generate/lite" or "edit/expand" && AspectRatio(size) is { } ratio)
            payload["aspect_ratio"] ??= JsonValue.Create(ratio);
        if (workflow.Sync) payload["sync"] ??= JsonValue.Create(true);

        using var client = clients.CreateClient(Id);
        var auth = Auth();
        var (initial, _) = await NativeHttp.PostJsonAsync(client, "image/" + model, payload, auth, headers, Json, ct);
        var current = initial;
        var target = String(current, "status_url");
        var requestId = String(current, "request_id");
        var polled = false;
        using var pollTimeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        pollTimeout.CancelAfter(TimeSpan.FromMinutes(2));
        while (FindImage(current) is null && !IsError(current) && Status(current) != "COMPLETED" && (target is not null || requestId is not null && Status(current) is "PENDING" or "IN_PROGRESS" or "PROCESSING"))
        {
            var poll = PollTarget(requestId, target);
            try
            {
                if (polled) await Task.Delay(TimeSpan.FromSeconds(1), pollTimeout.Token);
                (current, _) = await GetJobAsync(client, poll, auth, headers, pollTimeout.Token);
                polled = true;
            }
            catch (OperationCanceledException) when (!ct.IsCancellationRequested) { throw new TimeoutException("Bria image job timed out."); }
        }
        if (IsError(current)) throw new IOException(Error(current));
        var output = FindImage(current) ?? throw new JsonException("Bria returned no image output.");
        var (data, mime) = await DownloadAsync(output, "image/", ImageLimit, ct);
        var extension = mime switch { "image/jpeg" => "jpeg", "image/svg+xml" => "svg", _ => mime[6..] };
        return new OpenAIImagesResponse
        {
            Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), OutputFormat = extension,
            Data = [new() { B64Json = data }], AdditionalProperties = new()
            {
                ["providerMetadata"] = Metadata(polled ? JsonSerializer.SerializeToElement(new { submission = initial, result = current }, Json) : current)
            }
        };
    }

    private static string? FindImage(JsonElement body) => String(Property(body, "result"), "image_url") ?? String(body, "image_url");
    private static bool HasText(JsonObject body, string name)
        => body[name] is JsonValue value && value.TryGetValue<string>(out var text) && !string.IsNullOrWhiteSpace(text);

    private static JsonObject ImageOptions(Dictionary<string, JsonElement>? options, string? group)
    {
        var body = new JsonObject();
        var reserved = new HashSet<string>(["model", "image", "images", "files", "mask", "stream", "partial_images", "n", "response_format", "size", "api_token", "status_url", "request_id"], StringComparer.OrdinalIgnoreCase);
        void CopyFields(IEnumerable<KeyValuePair<string, JsonElement>> source)
        {
            foreach (var (key, value) in source)
                if (!reserved.Contains(key) && !Workflows.Values.Any(w => w.OptionGroup == key) && key is not ("providerOptions" or "provider_options" or Id))
                    body[key] = Copy(value);
        }
        if (options is null) return body;
        var scoped = options;
        foreach (var key in new[] { "providerOptions", "provider_options" })
            if (options.TryGetValue(key, out var wrapper) && Property(wrapper, Id) is { ValueKind: JsonValueKind.Object } selected) scoped = Fields(selected).ToDictionary();
        if (options.TryGetValue(Id, out var nested) && nested.ValueKind == JsonValueKind.Object) scoped = Fields(nested).ToDictionary();
        if (group is not null && scoped.TryGetValue(group, out var grouped))
        {
            if (grouped.ValueKind != JsonValueKind.Object) throw new ProviderRequestException($"Bria '{group}' options must be an object.");
            CopyFields(Fields(grouped));
        }
        CopyFields(scoped);
        return body;
    }

    private static IEnumerable<KeyValuePair<string, JsonElement>> Fields(JsonElement obj) => obj.EnumerateObject().Select(p => KeyValuePair.Create(p.Name, p.Value));
    private static string? AspectRatio(string? size)
    {
        if (size?.Split('x') is not [var w, var h] || !int.TryParse(w, out var width) || !int.TryParse(h, out var height) || width <= 0 || height <= 0) return null;
        var (a, b) = (width, height);
        while (b > 0) (a, b) = (b, a % b);
        return FormattableString.Invariant($"{width / a}:{height / a}");
    }

    private static string ImageValue(OpenAIImageReference input)
    {
        if (input.ImageUrl is not { Length: > 0 } image) throw new ProviderRequestException("Bria requires an image URL or inline image, not a file ID.");
        if (Uri.TryCreate(image, UriKind.Absolute, out var uri) && uri.Scheme is "http" or "https") return image;
        var comma = image.IndexOf(',');
        if (!image.StartsWith("data:image/", StringComparison.OrdinalIgnoreCase) || comma <= 0
            || !image[..comma].EndsWith(";base64", StringComparison.OrdinalIgnoreCase) || comma == image.Length - 1)
            throw new ProviderRequestException("Bria requires a URL or base64 image data URL.");
        if (image.Length - comma - 1 > 4L * ((ImageLimit + 2) / 3)) throw new ProviderRequestException("Image is too large.", 413);
        return image[(comma + 1)..];
    }

    private static async Task<string> InlineAsync(IFormFile upload, CancellationToken ct)
    {
        if (!upload.ContentType.StartsWith("image/", StringComparison.OrdinalIgnoreCase)) throw new ProviderRequestException("Bria requires image uploads.", 415);
        await using var file = upload.OpenReadStream();
        return await BoundedBase64Async(file, ImageLimit, ct);
    }

    private static Dictionary<string, JsonElement> Scoped(JsonElement body) => new() { [Id] = body };
    private static JsonElement Metadata(JsonElement body) => JsonSerializer.SerializeToElement(Scoped(body), Json);
    private static string? Status(JsonElement body) => String(body, "status")?.ToUpperInvariant();
    private static bool IsError(JsonElement body) => Status(body) is "ERROR" or "FAILED" or "UNKNOWN" or "NOT_FOUND" or "CANCELLED";
    private static string Error(JsonElement body) => String(Property(body, "error"), "message") ?? String(body, "message") ?? $"Bria job ended with status {Status(body)}.";

    private static bool SafeHttps(Uri uri) => uri.Scheme == Uri.UriSchemeHttps && uri.IsDefaultPort && uri.UserInfo.Length == 0 && uri.Fragment.Length == 0;

    private static Uri PollTarget(string? id, string? statusUrl)
    {
        if (statusUrl is not null && Uri.TryCreate(Api, statusUrl, out var uri) && SafeHttps(uri) && uri.Host == Api.Host
            && uri.AbsolutePath.StartsWith("/v2/status/", StringComparison.Ordinal) && SafeJobId(uri.AbsolutePath["/v2/status/".Length..])) return uri;
        if (!SafeJobId(id)) throw new JsonException("Bria returned no trusted status target.");
        return new Uri(Api, "status/" + id);
    }

    private static bool SafeJobId(string? id) => id is { Length: > 0 and <= 200 } && id.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_');

    private static async Task<(JsonElement Body, Dictionary<string, string?> Headers)> GetJobAsync(HttpClient client, Uri target,
        Action<HttpRequestHeaders> auth, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, target);
        auth(request.Headers);
        NativeHttp.ApplyHeaders(request, headers);
        using var response = await NativeHttp.SendAsync(client, request, ct);
        return (await NativeHttp.ReadAsync<JsonElement>(response, Json, ct), response.Headers.Concat(response.Content.Headers)
            .Where(h => !h.Key.Equals("Set-Cookie", StringComparison.OrdinalIgnoreCase)).GroupBy(h => h.Key, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(h => h.Key, h => (string?)string.Join(", ", h.SelectMany(x => x.Value)), StringComparer.OrdinalIgnoreCase));
    }

    private async Task<(string Data, string Mime)> DownloadAsync(string url, string prefix, int limit, CancellationToken ct)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out var target) || !SafeHttps(target) || target.HostNameType != UriHostNameType.Dns
            || !(target.Host == Api.Host || target.Host.EndsWith(".bria-api.com", StringComparison.Ordinal)
                || target.Host.EndsWith(".bria.ai", StringComparison.Ordinal)
                || (configuration["Providers:bria:DeliveryHosts"] ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Contains(target.IdnHost, StringComparer.OrdinalIgnoreCase)))
            throw new IOException("Untrusted Bria delivery host.");
        using var client = clients.CreateClient(DeliveryClient);
        using var request = new HttpRequestMessage(HttpMethod.Get, target);
        using var response = await NativeHttp.SendAsync(client, request, ct);
        if (response.Content.Headers.ContentLength > limit) throw new IOException("Bria output is too large.");
        var mime = response.Content.Headers.ContentType?.MediaType;
        if (mime is null or "application/octet-stream") mime = Path.GetExtension(target.AbsolutePath).ToLowerInvariant() switch
        {
            ".jpg" or ".jpeg" => "image/jpeg", ".webp" => "image/webp", ".gif" => "image/gif", ".png" => "image/png", ".avif" => "image/avif",
            ".webm" => "video/webm", ".mov" => "video/quicktime", ".mp4" => "video/mp4", _ => prefix == "image/" ? "image/png" : "video/mp4"
        };
        if (!mime.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) && (prefix != "video/" || mime != "image/gif"))
            throw new IOException("Bria returned an unexpected media type.");
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        return (await BoundedBase64Async(stream, limit, ct), mime.ToLowerInvariant());
    }
}
