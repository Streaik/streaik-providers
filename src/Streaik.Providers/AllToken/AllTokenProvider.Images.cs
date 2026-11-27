using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class AllTokenProvider
{
    private static readonly string[] ImageFields =
        ["model", "prompt", "size", "n", "quality", "output_format", "output_compression", "background", "moderation", "user"];
    private static readonly HashSet<string> ReservedImageFields = new(ImageFields.Concat(
        ["image_upload_id", "mask_upload_id", "response_format", "stream", "partial_images", "images", "files", "mask", "aspectRatio", "aspect_ratio"]), StringComparer.OrdinalIgnoreCase);

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => RunImageAsync("images/generations/async", ImagePayload(request, request.AdditionalProperties), headers, ct);

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var payload = ImagePayload(request, request.AdditionalProperties);
        var file = request.ImageFiles?.FirstOrDefault();
        var reference = request.Images?.FirstOrDefault();
        if (file is null && reference is null) throw new ProviderRequestException("AllToken image edits require a source image.");
        if (file is not null && reference is not null || request.MaskFile is not null && request.Mask is not null)
            throw new ProviderRequestException("Choose either uploaded files or inline image references.");

        using var client = CreateClient();
        using var assets = CreateClient(AssetClient);
        payload["image_upload_id"] = await UploadImageAsync(client, assets, file, reference, "image_edit_source", headers, ct);
        if (request.MaskFile is not null || request.Mask is not null)
            payload["mask_upload_id"] = await UploadImageAsync(client, assets, request.MaskFile, request.Mask, "image_edit_mask", headers, ct);
        return await RunImageAsync("images/edits", payload, headers, ct);
    }

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageGenerationRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        if (request.N is > 1) throw new ProviderRequestException("Image streaming requires a single output.");
        yield return NativeMediaOptions.CompletedImage(await GenerateImageAsync(request, headers, ct), edit: false);
    }

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageEditRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        if (request.N is > 1) throw new ProviderRequestException("Image streaming requires a single output.");
        yield return NativeMediaOptions.CompletedImage(await EditImageAsync(request, headers, ct), edit: true);
    }

    private static JsonObject ImagePayload<T>(T request, IReadOnlyDictionary<string, JsonElement>? options)
    {
        var payload = new JsonObject();
        foreach (var (name, value) in NativeMediaOptions.Read(Id, options, ReservedImageFields)) payload[name] = Copy(value);
        var typed = ToObject(request);
        foreach (var field in ImageFields)
            if (typed[field] is { } value) payload[field] = value.DeepClone();
        return payload;
    }

    private async Task<OpenAIImagesResponse> RunImageAsync(string endpoint, JsonObject payload,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromMinutes(5));
        var token = deadline.Token;
        using var client = CreateClient();
        using var assets = CreateClient(AssetClient);
        try
        {
            var (body, _) = await NativeHttp.PostJsonAsync(client, endpoint, payload, Bearer(ApiKey()), headers, Json, token);
            if (Status(body) != "completed" && !Failed(Status(body)))
            {
                var path = TaskPath("images/generations/", String(body, "id"));
                while (true)
                {
                    (body, _) = await ReadTaskAsync(client, path, headers, token);
                    if (Status(body) == "completed" || Failed(Status(body))) break;
                    await Task.Delay(TimeSpan.FromSeconds(2), token);
                }
            }
            if (Status(body) != "completed") throw new IOException("AllToken image job failed.");
            return await ReadImagesAsync(body, payload, assets, token);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested && deadline.IsCancellationRequested)
        {
            throw new TimeoutException("AllToken image generation timed out.");
        }
    }

    private static async Task<OpenAIImagesResponse> ReadImagesAsync(JsonElement body, JsonObject payload, HttpClient assets, CancellationToken ct)
    {
        var result = new List<OpenAIImageData>();
        var format = String(body, "output_format") ?? payload["output_format"]?.GetValue<string>();
        var fallback = format switch { "jpeg" or "jpg" => "image/jpeg", "webp" => "image/webp", _ => "image/png" };
        var raw = ToObject(body);
        foreach (var item in Array(Property(body, "data")))
        {
            var b64 = String(item, "b64_json");
            var mime = String(item, "mime_type");
            if (string.IsNullOrWhiteSpace(b64) && String(item, "r2_url") is { } url)
                (b64, mime) = await PublicMediaDelivery.DownloadAsync(assets, url, fallback, ImageLimit, ct);
            if (string.IsNullOrWhiteSpace(b64)) continue;
            if (b64.Length > (ImageLimit + 2) / 3 * 4) throw new ProviderRequestException("Image exceeds the gateway's inline size limit.", 413);
            format ??= mime switch { "image/jpeg" => "jpeg", "image/webp" => "webp", _ => null };
            result.Add(new() { B64Json = b64, RevisedPrompt = String(item, "revised_prompt") });
        }
        if (result.Count == 0) throw new JsonException("AllToken completed without images.");
        if (raw["data"] is JsonArray data)
            foreach (var entry in data.OfType<JsonObject>())
            {
                entry.Remove("b64_json");
                entry.Remove("r2_url");
            }
        var created = Property(body, "created");
        var usage = Property(body, "usage");
        return new()
        {
            Created = created.ValueKind == JsonValueKind.Number && created.TryGetInt64(out var unix) ? unix : Timestamp(body).ToUnixTimeSeconds(),
            Data = result, OutputFormat = format ?? "png",
            Background = String(body, "background") ?? payload["background"]?.GetValue<string>(),
            Quality = String(body, "quality") ?? payload["quality"]?.GetValue<string>(),
            Size = String(body, "size") ?? payload["size"]?.GetValue<string>(),
            Usage = usage.ValueKind == JsonValueKind.Object ? usage.Deserialize<OpenAIImageUsage>(Json) : null,
            AdditionalProperties = new() { ["providerMetadata"] = JsonSerializer.SerializeToElement(NativeMediaOptions.Metadata(Id, raw), Json) }
        };
    }

    private async Task<string> UploadImageAsync(HttpClient client, HttpClient assets, IFormFile? file, OpenAIImageReference? reference,
        string purpose, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        byte[] bytes;
        string mime;
        if (file is not null)
        {
            if (file.Length > ImageLimit) throw new ProviderRequestException("Image exceeds the gateway's inline size limit.", 413);
            await using var stream = file.OpenReadStream();
            using var buffer = new MemoryStream((int)file.Length);
            await stream.CopyToAsync(buffer, ct);
            bytes = buffer.ToArray();
            mime = MediaTypeHeaderValue.TryParse(file.ContentType, out var type) ? type.MediaType ?? "image/png" : "image/png";
        }
        else
        {
            var data = reference?.ImageUrl;
            var separator = data?.IndexOf(";base64,", StringComparison.OrdinalIgnoreCase) ?? -1;
            if (data is null || !data.StartsWith("data:image/", StringComparison.OrdinalIgnoreCase) || separator is < 12 or > 40)
                throw new ProviderRequestException("AllToken image edits require an uploaded image or an inline base64 image.");
            if (data.Length - separator - 8 > (ImageLimit + 2) / 3 * 4)
                throw new ProviderRequestException("Image exceeds the gateway's inline size limit.", 413);
            mime = data[5..separator].ToLowerInvariant();
            try { bytes = Convert.FromBase64String(data[(separator + 8)..]); }
            catch (FormatException) { throw new ProviderRequestException("Invalid base64 image."); }
        }
        if (bytes.Length == 0) throw new ProviderRequestException("Missing image content.");
        if (mime is not ("image/png" or "image/jpeg" or "image/webp" or "image/gif" or "image/avif"))
            throw new ProviderRequestException("Unsupported AllToken upload image type.", 415);

        var presign = new { purpose, content_type = mime, content_length = bytes.Length,
            content_md5 = Convert.ToBase64String(MD5.HashData(bytes)), checksum_sha256 = Convert.ToHexStringLower(SHA256.HashData(bytes)) };
        var (body, _) = await NativeHttp.PostJsonAsync(client, "uploads/presign", presign, Bearer(ApiKey()), headers, Json, ct);
        var id = String(body, "upload_id");
        if (string.IsNullOrWhiteSpace(id)) throw new JsonException("AllToken returned no upload ID.");
        using var request = new HttpRequestMessage(HttpMethod.Put, PublicMediaDelivery.RequirePublicHttpsUri(String(body, "upload_url")))
            { Content = new ByteArrayContent(bytes) };
        if (Property(body, "required_headers") is { ValueKind: JsonValueKind.Object } required)
            foreach (var field in required.EnumerateObject())
            {
                if (field.Value.ValueKind != JsonValueKind.String || field.Value.GetString() is not { } value || value.Contains('\r') || value.Contains('\n')) continue;
                var name = field.Name.ToLowerInvariant();
                if (name is "content-type" or "content-md5") request.Content.Headers.TryAddWithoutValidation(field.Name, value);
                if (name.StartsWith("x-amz-checksum-", StringComparison.Ordinal) || name.StartsWith("x-amz-meta-", StringComparison.Ordinal)
                    || name is "x-amz-sdk-checksum-algorithm" or "x-amz-content-sha256")
                    request.Headers.TryAddWithoutValidation(field.Name, value);
            }
        request.Content.Headers.ContentType ??= new(mime);
        using var response = await NativeHttp.SendAsync(assets, request, ct);
        return id;
    }
}
