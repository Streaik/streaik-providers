using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Streaik.OpenAI.Models;

namespace Streaik.Providers;

internal static class MediaHttp
{
    public static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web) { DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull };
    public static string RequiredKey(IProviderCredentials credentials, string id)
        => credentials.GetApiKey(id) is { } key && !string.IsNullOrWhiteSpace(key) ? key : throw new ProviderCredentialException(id);

    public static async Task<HttpResponseMessage> SendAsync(HttpClient client, HttpMethod method, string path, HttpContent? content,
        string scheme, string key, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var request = new HttpRequestMessage(method, path) { Content = content };
        request.Headers.Authorization = new(scheme, key);
        NativeHttp.ApplyHeaders(request, headers);
        return await NativeHttp.SendAsync(client, request, ct);
    }

    public static async Task<T> JsonAsync<T>(HttpClient client, HttpMethod method, string path, HttpContent? content,
        string scheme, string key, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var response = await SendAsync(client, method, path, content, scheme, key, headers, ct);
        return await NativeHttp.ReadAsync<T>(response, Json, ct);
    }

    public static JsonNode? Copy(JsonElement element) => JsonNode.Parse(element.GetRawText());
    public static JsonObject ToObject<T>(T value) => (JsonObject)JsonSerializer.SerializeToNode(value, Json)!;
    public static string? String(JsonElement value, string name)
        => value.ValueKind == JsonValueKind.Object && value.TryGetProperty(name, out var property) && property.ValueKind == JsonValueKind.String
            ? property.GetString() : null;
    public static JsonElement Property(JsonElement value, string name) => value.ValueKind == JsonValueKind.Object && value.TryGetProperty(name, out var p) ? p : default;
    public static IEnumerable<JsonElement> Array(JsonElement value) => value.ValueKind == JsonValueKind.Array ? value.EnumerateArray() : [];

    public static async Task<string> BoundedBase64Async(Stream input, int limit, CancellationToken ct)
    {
        using var buffer = new MemoryStream();
        var chunk = new byte[32 * 1024];
        while (await input.ReadAsync(chunk, ct) is > 0 and var count)
        {
            if (buffer.Length + count > limit) throw new ProviderRequestException("Media exceeds the gateway's inline size limit.", 413);
            buffer.Write(chunk, 0, count);
        }
        if (buffer.Length == 0) throw new JsonException("Missing media content.");
        return Convert.ToBase64String(buffer.GetBuffer(), 0, (int)buffer.Length);
    }

    public static MultipartFormDataContent ImageMultipart(OpenAIImageEditRequest request, bool inlineReferences, string imageField, string maskField)
    {
        var form = new MultipartFormDataContent();
        try
        {
            var fields = ToObject(request);
            fields.Remove("images"); fields.Remove("mask"); fields.Remove("stream"); fields.Remove("partial_images");
            AddFields(form, fields);
            foreach (var file in request.ImageFiles ?? []) AddFile(form, imageField, file);
            if (request.MaskFile is { } mask) AddFile(form, maskField, mask);
            if (inlineReferences)
            {
                foreach (var image in request.Images ?? []) AddDataImage(form, imageField, image.ImageUrl);
                if (request.Mask is not null) AddDataImage(form, maskField, request.Mask.ImageUrl);
            }
            else if (request.Images is { Length: > 0 } || request.Mask is not null)
                throw new ProviderRequestException("Do not mix file uploads and JSON image references.");
            return form;
        }
        catch { form.Dispose(); throw; }
    }

    public static void AddFields(MultipartFormDataContent form, JsonObject fields)
    {
        foreach (var (name, value) in fields)
            if (value is not null)
                form.Add(new StringContent(value is JsonValue item && item.TryGetValue<string>(out var text) ? text : value.ToJsonString()), name);
    }

    public static void AddFile(MultipartFormDataContent form, string field, IFormFile file)
        => AddFile(form, field, file.OpenReadStream(), file.ContentType, file.FileName);

    public static void AddFile(MultipartFormDataContent form, string field, Stream stream, string mediaType, string fileName)
    {
        var content = new StreamContent(stream);
        content.Headers.ContentType = MediaTypeHeaderValue.TryParse(mediaType, out var type) ? type : new("application/octet-stream");
        var name = new string(Path.GetFileName(fileName).Where(ch => char.IsAsciiLetterOrDigit(ch) || ch is '.' or '-' or '_').Take(120).ToArray());
        try { form.Add(content, field, string.IsNullOrEmpty(name) ? "upload" : name); }
        catch { content.Dispose(); throw; }
    }

    public static void AddDataImage(MultipartFormDataContent form, string field, string? dataUrl)
    {
        if (dataUrl is null || !dataUrl.StartsWith("data:image/", StringComparison.Ordinal))
            throw new ProviderRequestException("This provider requires an uploaded file or inline base64 image (no remote image fetch).", 422);
        var comma = dataUrl.IndexOf(',');
        if (comma is < 12 or > 64 || !dataUrl[..comma].EndsWith(";base64", StringComparison.Ordinal))
            throw new ProviderRequestException("Invalid image data URL.");
        var mime = dataUrl[5..(comma - 7)];
        var extension = mime switch { "image/png" => "png", "image/jpeg" => "jpg", "image/webp" => "webp", "image/gif" => "gif", _ => null };
        if (extension is null) throw new ProviderRequestException("Unsupported inline image type.", 415);
        try { AddFile(form, field, new MemoryStream(Convert.FromBase64String(dataUrl[(comma + 1)..]), writable: false), mime, $"{field.TrimEnd('[', ']')}.{extension}"); }
        catch (FormatException) { throw new ProviderRequestException("Invalid base64 image."); }
    }
}
