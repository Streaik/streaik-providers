using System.Globalization;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class LumaAIRegistration
{
    public static IServiceCollection AddLumaAI(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, "lumaai", "https://agents.lumalabs.ai/");
        services.AddHttpClient("lumaai-delivery").ConfigurePrimaryHttpMessageHandler(() =>
            new SocketsHttpHandler { UseCookies = false, AllowAutoRedirect = false });
        return services.AddProvider<LumaAIProvider>("lumaai");
    }
}

internal sealed class LumaAIProvider(IHttpClientFactory clients, IProviderCredentials credentials, IConfiguration config)
    : IModelListingProvider, IImageModelRoutingProvider, IVideoProvider
{
    private const string Id = "lumaai";
    private const int ImageLimit = 25 * 1024 * 1024;
    private const int VideoLimit = 64 * 1024 * 1024;
    private static readonly HashSet<string> Reserved = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "prompt", "type", "source", "image_ref", "images", "mask", "stream", "partial_images", "response_format",
        "authorization", "headers", "api_key", "apiKey", "api-key", "url", "endpoint", "providerOptions", Id
    };

    public IAsyncEnumerable<Model> ListModelsAsync(CancellationToken ct) => StaticModelListing.ListAsync(Id, credentials.GetApiKey(Id), ct);
    private static bool IsImageModel(string model) => model is "uni-1" or "uni-1-max";
    public Task<bool> IsImageModelAsync(string model, CancellationToken ct) => Task.FromResult(IsImageModel(model));

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => RunImageAsync(request.Model, request.Prompt, request.N, request.Size,
            request.AdditionalProperties, [], false, headers, ct);

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var references = new List<object>();
        var remaining = ImageLimit;
        foreach (var file in request.ImageFiles ?? [])
        {
            ValidateImageType(file.ContentType);
            if (file.Length > remaining) throw new ProviderRequestException("Luma image inputs exceed the inline size limit.", 413);
            await using var stream = file.OpenReadStream();
            var data = await BoundedBase64Async(stream, remaining, ct);
            remaining -= Base64Size(data);
            references.Add(new { data, media_type = file.ContentType });
        }
        foreach (var image in request.Images ?? [])
        {
            var uri = image.ImageUrl;
            if (uri is null || !uri.StartsWith("data:image/", StringComparison.OrdinalIgnoreCase))
                throw new ProviderRequestException("Luma image editing requires uploads or inline image data URLs.", 422);
            var comma = uri.IndexOf(',');
            if (comma is < 13 or > 128 || !uri[..comma].EndsWith(";base64", StringComparison.OrdinalIgnoreCase))
                throw new ProviderRequestException("Invalid Luma image data URL.");
            references.Add(ImageReference(uri[(comma + 1)..], uri[5..(comma - 7)], ref remaining));
        }
        if (references.Count == 0) throw new ProviderRequestException("Luma image editing requires an image.");
        return await RunImageAsync(request.Model, request.Prompt, request.N, request.Size, request.AdditionalProperties,
            references, request.Mask is not null || request.MaskFile is not null, headers, ct);
    }

    private async Task<OpenAIImagesResponse> RunImageAsync(string model, string prompt, int? count, string? size,
        Dictionary<string, JsonElement>? options, List<object> references, bool mask, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!IsImageModel(model)) throw new ProviderRequestException("Unsupported Luma image model.", 422);
        if (string.IsNullOrWhiteSpace(prompt)) throw new ProviderRequestException("An image prompt is required.");
        var body = Options(options, flat: true);
        var warnings = new List<object>();
        if (count > 1) Warn(warnings, "n");
        if (!string.IsNullOrWhiteSpace(size)) Warn(warnings, "size");
        if (body.Remove("seed")) Warn(warnings, "seed");
        if (mask) Warn(warnings, "mask");
        body.Remove("video");
        body["model"] = model;
        body["prompt"] = prompt;
        body["type"] = references.Count > 0 ? "image_edit" : "image";
        if (references.Count > 0) body["source"] = JsonSerializer.SerializeToNode(references[0], Json);
        if (references.Count > 1) body["image_ref"] = JsonSerializer.SerializeToNode(references.Skip(1), Json);

        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromMinutes(5));
        var token = deadline.Token;
        try
        {
            using var client = clients.CreateClient(Id);
            var result = await StartAsync(client, body, headers, token);
            var id = GenerationId(result);
            while (State(result) is not ("completed" or "failed" or "canceled" or "cancelled"))
            {
                result = await StatusAsync(client, id, headers, token);
                if (State(result) is "completed" or "failed" or "canceled" or "cancelled") break;
                await Task.Delay(TimeSpan.FromSeconds(2), token);
            }
            if (State(result) != "completed") throw new IOException("Luma image generation did not complete successfully.");
            var (data, mime) = await DownloadAsync(Output(result, "image"), "image", ImageLimit, token);
            return new() { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Data = [new() { B64Json = data }], OutputFormat = mime[6..],
                AdditionalProperties = new() { ["providerMetadata"] = JsonSerializer.SerializeToElement(Metadata(result), Json),
                    ["warnings"] = JsonSerializer.SerializeToElement(warnings, Json) } };
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested && deadline.IsCancellationRequested)
        { throw new TimeoutException("Luma image generation timed out."); }
    }

    public async Task<VideoOperationStartResult> StartVideoAsync(VideoRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Model != "ray-3.2") throw new ProviderRequestException("Unsupported Luma video model.", 422);
        if (string.IsNullOrWhiteSpace(request.Prompt) && request.Image is null)
            throw new ProviderRequestException("A prompt or an image start frame is required.");
        var body = Options(request.ProviderOptions);
        var video = body["video"] switch
        {
            null => new JsonObject(),
            JsonObject node => (JsonObject)node.DeepClone(),
            _ => throw new ProviderRequestException("Luma video options must be an object.")
        };
        if (body.Remove("loop", out var loop)) video["loop"] = loop;
        if (request.Resolution is not null) video["resolution"] = request.Resolution;
        if (request.Duration is { } duration) video["duration"] = duration.ToString(CultureInfo.InvariantCulture) + "s";
        if (request.Image is { } image)
        {
            var remaining = ImageLimit;
            video["start_frame"] = JsonSerializer.SerializeToNode(ImageReference(image.Data, image.MediaType, ref remaining), Json);
        }
        body["model"] = request.Model;
        body["type"] = "video";
        if (!string.IsNullOrWhiteSpace(request.Prompt)) body["prompt"] = request.Prompt;
        if (request.AspectRatio is not null) body["aspect_ratio"] = request.AspectRatio;
        body["video"] = video;
        var warnings = new List<object>();
        if (request.N > 1) Warn(warnings, "n");
        if (request.Seed is not null) Warn(warnings, "seed");
        if (request.Fps is not null) Warn(warnings, "fps");
        if (request.GenerateAudio is not null) Warn(warnings, "generateAudio");
        if (request.InputReferences?.Any() == true) Warn(warnings, "inputReferences");
        if (request.FrameImages?.Any() == true) Warn(warnings, "frameImages");
        using var client = clients.CreateClient(Id);
        var result = await StartAsync(client, body, headers, ct);
        return new() { Operation = GenerationId(result), Warnings = warnings, ProviderMetadata = Metadata(result),
            Response = new() { Timestamp = DateTime.UtcNow, ModelId = request.Model } };
    }

    public async Task<VideoOperationStatusResult> GetVideoAsync(string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        var result = await StatusAsync(client, operation, headers, ct);
        var response = new HeaderResponseData { ModelId = String(result, "model") ?? "ray-3.2", Timestamp = DateTime.UtcNow };
        var metadata = Metadata(result);
        if (State(result) is "failed" or "canceled" or "cancelled")
            return new VideoOperationErrorResult { Error = "Luma video generation did not complete successfully.", ProviderMetadata = metadata, Response = response };
        if (State(result) != "completed") return new VideoOperationPendingResult { ProviderMetadata = metadata, Response = response };
        if (OutputUrl(result, "video") is null)
            return new VideoOperationErrorResult { Error = "Luma returned no video output.", ProviderMetadata = metadata, Response = response };
        var (data, mime) = await DownloadAsync(Output(result, "video"), "video", VideoLimit, ct);
        return new VideoOperationCompletedResult { Videos = [new() { Type = "base64", MediaType = mime, Data = data }],
            ProviderMetadata = metadata, Response = response };
    }

    private Task<JsonElement> StartAsync(HttpClient client, JsonObject body, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => JsonAsync<JsonElement>(client, HttpMethod.Post, "v1/generations", JsonContent.Create(body, options: Json),
            "Bearer", RequiredKey(credentials, Id), headers, ct);

    private Task<JsonElement> StatusAsync(HttpClient client, string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!ValidId(operation)) throw new ProviderRequestException("Invalid Luma generation ID.");
        return JsonAsync<JsonElement>(client, HttpMethod.Get, $"v1/generations/{operation}", null, "Bearer", RequiredKey(credentials, Id), headers, ct);
    }

    private static bool ValidId(string? id) => id is { Length: > 0 and <= 256 } && id.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_');
    private static string GenerationId(JsonElement root) => String(root, "id") is { } id && ValidId(id) ? id
        : throw new JsonException("Luma returned an invalid generation ID.");
    private static string? State(JsonElement root) => String(root, "state")?.Trim().ToLowerInvariant();
    private static Dictionary<string, JsonElement> Metadata(JsonElement root) => new() { [Id] = root };
    private static void Warn(List<object> warnings, string feature) => warnings.Add(new { type = "unsupported", feature });

    private static JsonObject Options(Dictionary<string, JsonElement>? extras, bool flat = false)
    {
        var body = new JsonObject();
        if (extras?.TryGetValue(Id, out var native) == true && native.ValueKind != JsonValueKind.Null)
        {
            if (native.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Luma provider options must be an object.");
            foreach (var field in native.EnumerateObject()) if (!Reserved.Contains(field.Name)) body[field.Name] = Copy(field.Value);
        }
        if (flat && extras is not null)
            foreach (var (name, value) in extras) if (!Reserved.Contains(name)) body[name] = Copy(value);
        return body;
    }

    private static void ValidateImageType(string? type)
    {
        if (type is not ("image/png" or "image/jpeg" or "image/webp" or "image/gif" or "image/avif"))
            throw new ProviderRequestException("Unsupported Luma image input type.", 415);
    }

    private static int Base64Size(string data) => data.Length / 4 * 3 - (data.EndsWith("==", StringComparison.Ordinal) ? 2 : data.EndsWith('=') ? 1 : 0);

    private static object ImageReference(string? data, string? mime, ref int remaining)
    {
        if (data?.StartsWith("data:", StringComparison.OrdinalIgnoreCase) == true)
        {
            var comma = data.IndexOf(',');
            if (comma is < 13 or > 128 || !data[..comma].EndsWith(";base64", StringComparison.OrdinalIgnoreCase))
                throw new ProviderRequestException("Invalid Luma image data URL.");
            mime = data[5..(comma - 7)];
            data = data[(comma + 1)..];
        }
        ValidateImageType(mime);
        if (string.IsNullOrWhiteSpace(data)) throw new ProviderRequestException("Luma requires inline image data.");
        if (data.Length > (remaining + 2L) / 3 * 4) throw new ProviderRequestException("Luma image inputs exceed the inline size limit.", 413);
        if (!System.Buffers.Text.Base64.IsValid(data, out var size) || size == 0) throw new ProviderRequestException("Invalid Luma image base64.");
        if (size > remaining) throw new ProviderRequestException("Luma image inputs exceed the inline size limit.", 413);
        remaining -= size;
        return new { data, media_type = mime };
    }

    private static string? OutputUrl(JsonElement root, string kind) => Array(Property(root, "output"))
        .Where(item => String(item, "type") is not { Length: > 0 } type || type.Equals(kind, StringComparison.OrdinalIgnoreCase))
        .Select(item => String(item, "url")).FirstOrDefault(url => !string.IsNullOrWhiteSpace(url));

    private Uri Output(JsonElement root, string kind)
    {
        if (!Uri.TryCreate(OutputUrl(root, kind), UriKind.Absolute, out var target) || target.Scheme != Uri.UriSchemeHttps
            || !target.IsDefaultPort || target.UserInfo.Length != 0 || target.Fragment.Length != 0 || target.HostNameType != UriHostNameType.Dns
            || !(target.IdnHost == "lumalabs.ai" || target.IdnHost.EndsWith(".lumalabs.ai", StringComparison.Ordinal)
                || target.IdnHost.EndsWith(".cdn-luma.com", StringComparison.Ordinal)
                || (config["Providers:lumaai:DeliveryHosts"] ?? "").Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
                    .Contains(target.IdnHost, StringComparer.OrdinalIgnoreCase)))
            throw new IOException("Luma did not return a trusted media URL.");
        return target;
    }

    private async Task<(string Data, string Mime)> DownloadAsync(Uri target, string kind, int limit, CancellationToken ct)
    {
        using var client = clients.CreateClient("lumaai-delivery");
        using var message = new HttpRequestMessage(HttpMethod.Get, target);
        // The signed asset URL authenticates the download. Never attach provider or caller headers.
        using var response = await NativeHttp.SendAsync(client, message, ct);
        var mime = response.Content.Headers.ContentType?.MediaType?.ToLowerInvariant();
        if (mime is null or "application/octet-stream") mime = Path.GetExtension(target.AbsolutePath).ToLowerInvariant() switch
        {
            ".png" => "image/png", ".webp" => "image/webp", ".gif" => "image/gif", ".avif" => "image/avif",
            ".webm" => "video/webm", ".mov" => "video/quicktime", _ => kind == "video" ? "video/mp4" : "image/jpeg"
        };
        if (!mime.StartsWith(kind + "/", StringComparison.Ordinal) || mime == "image/svg+xml") throw new IOException("Unexpected Luma output content type.");
        if (response.Content.Headers.ContentLength > limit) throw new IOException("Luma output exceeds the inline size limit.");
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        return (await BoundedBase64Async(stream, limit, ct), mime);
    }
}
