using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.Catalog;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class VanceAIProviderRegistration
{
    public static IServiceCollection AddVanceAI(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, VanceAIProvider.Id, "https://vanceai.com/api/v1/");
        PublicMediaDelivery.AddClient(services, VanceAIProvider.TransferClient);
        return services.AddProvider<VanceAIProvider>(VanceAIProvider.Id);
    }
}

internal sealed class VanceAIProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IImageModelRoutingProvider, IVideoProvider
{
    internal const string Id = "vanceai";
    internal const string TransferClient = "vanceai-transfer";
    private const int MaxBytes = 25 * 1024 * 1024;
    private static readonly HashSet<string> TransportFields = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "tool", "config", "output_format", "outputFormat", "image", "images", "image_url", "imageUrl", "source_url", "sourceUrl",
        "upload_id", "uploadId", "file_name", "fileName", "content_type", "contentType", "file_size", "mask", "n", "prompt",
        "stream", "partial_images", "response_format"
    };
    private static readonly Lazy<Dictionary<string, Model>> Tools = new(() =>
        (ProviderCatalog.ReadModels(Id)?.Deserialize<ModelResponse>(Json)?.Data ?? throw new InvalidOperationException("Missing VanceAI tools."))
            .ToDictionary(item => item.Id[(Id.Length + 1)..], StringComparer.OrdinalIgnoreCase));

    public IAsyncEnumerable<Model> ListModelsAsync(CancellationToken ct) => StaticModelListing.ListAsync(Id, credentials.GetApiKey(Id), ct);
    public Task<bool> IsImageModelAsync(string model, CancellationToken ct) => Task.FromResult(IsTool(model, "image"));

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => throw new ProviderRequestException("VanceAI requires a source image. Use the image edits endpoint or supply an image in the SDK request.");

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var tool = RequireTool(request.Model, "image");
        if (request.N is not (null or 1) || request.Mask is not null || request.MaskFile is not null || request.Markup is not null
            || (request.Images?.Length ?? 0) + (request.ImageFiles?.Length ?? 0) != 1)
            throw new ProviderRequestException("VanceAI requires exactly one source image and does not support masks, markup, or batches.");
        var options = NativeMediaOptions.Read(Id, request.AdditionalProperties, NoReservedFields);
        var config = Config(options);
        if (tool == "custom")
        {
            if (string.IsNullOrWhiteSpace(request.Prompt)) throw new ProviderRequestException("VanceAI custom requires a prompt.");
            config["prompt"] = request.Prompt;
        }
        var format = (request.OutputFormat ?? Option(options, "output_format", "outputFormat") ?? "jpg").ToLowerInvariant();
        var fallback = format switch { "jpg" or "jpeg" => "image/jpeg", "png" => "image/png", "webp" => "image/webp",
            _ => throw new ProviderRequestException("VanceAI output format must be jpg, jpeg, png, or webp.") };
        if (format == "jpeg") format = "jpg";

        var reference = request.Images?.Single();
        if (reference?.FileId is not null) throw new ProviderRequestException("VanceAI does not accept image file IDs.");
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromMinutes(2));
        var token = deadline.Token;
        using var client = clients.CreateClient(Id);
        var auth = NativeHttp.Bearer(RequiredKey(credentials, Id));
        using var create = JobRequest(headers, auth);
        if (SourceUrl(reference?.ImageUrl) is { } imageUrl)
            create.Content = JsonContent.Create(new { image_url = imageUrl, tool, config, output_format = format }, options: Json);
        else
        {
            var form = new MultipartFormDataContent();
            create.Content = form;
            if (request.ImageFiles?.Single() is { } file)
            {
                if (file.Length <= 0 || file.Length > MaxBytes) throw new ProviderRequestException("Invalid VanceAI image upload size.", 413);
                if (!file.ContentType.StartsWith("image/", StringComparison.OrdinalIgnoreCase)) throw new ProviderRequestException("VanceAI requires an image upload.", 415);
                AddFile(form, "image", file);
            }
            else
            {
                var (bytes, mime) = ReadInline(reference?.ImageUrl, "image/", "image/jpeg");
                AddFile(form, "image", new MemoryStream(bytes, writable: false), mime, "image." + ImageFormat(mime));
            }
            form.Add(new StringContent(tool), "tool");
            form.Add(new StringContent(config.ToJsonString(Json)), "config");
            form.Add(new StringContent(format), "output_format");
        }

        try
        {
            var job = await ReadJobAsync(client, create, token);
            while (job.Status != "succeeded")
            {
                if (Failed(job)) throw new IOException(Failure(job));
                job = await PollJobAsync(client, job.Id, auth, headers, token);
                if (job.Status != "succeeded" && !Failed(job)) await Task.Delay(TimeSpan.FromSeconds(2), token);
            }
            while (true)
            {
                if (await DownloadAsync(client, job.Id, fallback, auth, headers, token) is { } result)
                    return new OpenAIImagesResponse
                    {
                        Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), OutputFormat = ImageFormat(result.Mime),
                        Data = [new OpenAIImageData { B64Json = result.Data }],
                        AdditionalProperties = new() { ["providerMetadata"] = JsonSerializer.SerializeToElement(Metadata(job), Json) }
                    };
                await Task.Delay(TimeSpan.FromSeconds(1), token);
            }
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested && deadline.IsCancellationRequested)
        {
            throw new ProviderRequestException("VanceAI image processing timed out.", 504);
        }
    }

    public async Task<VideoOperationStartResult> StartVideoAsync(VideoRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var tool = RequireTool(request.Model, "video");
        var references = request.InputReferences?.ToArray() ?? [];
        var source = references.Append(request.Image).Concat((request.FrameImages ?? []).Select(frame => frame.Image))
            .FirstOrDefault(file => file?.MediaType?.StartsWith("video/", StringComparison.OrdinalIgnoreCase) == true)
            // The SDK's URL reference contract has no media type. These tools only take video inputs.
            ?? references.OfType<VideoFileUrl>().FirstOrDefault(file => file.MediaType is null)
            ?? throw new ProviderRequestException("VanceAI requires a video input reference.");
        if (source.Type is "file_id" or "fileId") throw new ProviderRequestException("VanceAI does not accept video file IDs.");
        var options = NativeMediaOptions.Read(Id, request.ProviderOptions, NoReservedFields);
        var config = Config(options);
        if (tool == "video_upscale" && request.Fps is { } fps) config["fps"] = fps;
        var warnings = new List<object>();
        Warn("prompt", !string.IsNullOrWhiteSpace(request.Prompt));
        Warn("n", request.N is > 1);
        Warn("seed", request.Seed.HasValue);
        Warn("resolution", request.Resolution is not null);
        Warn("aspectRatio", request.AspectRatio is not null);
        Warn("duration", request.Duration.HasValue);
        Warn("generateAudio", request.GenerateAudio.HasValue);
        Warn("fps", request.Fps.HasValue && tool != "video_upscale");

        using var client = clients.CreateClient(Id);
        var auth = NativeHttp.Bearer(RequiredKey(credentials, Id));
        var payload = new JsonObject { ["tool"] = tool, ["config"] = config };
        if (SourceUrl(source is VideoFileUrl url ? url.Url : source.Data) is { } sourceUrl) payload["source_url"] = sourceUrl;
        else payload["upload_id"] = await UploadAsync(client, source, options, auth, headers, ct);
        using var create = JobRequest(headers, auth);
        create.Content = JsonContent.Create(payload, options: Json);
        var job = await ReadJobAsync(client, create, ct);
        return new() { Operation = $"{tool}/{job.Id}", Warnings = warnings, ProviderMetadata = Metadata(job), Response = Response(job, tool) };

        void Warn(string feature, bool unsupported) { if (unsupported) warnings.Add(new { type = "unsupported", feature }); }
    }

    public async Task<VideoOperationStatusResult> GetVideoAsync(string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var parts = operation.Split('/');
        if (parts.Length != 2 || !IsTool(parts[0], "video") || !ValidId(parts[1]))
            throw new ProviderRequestException("Invalid VanceAI video operation.");
        using var client = clients.CreateClient(Id);
        var auth = NativeHttp.Bearer(RequiredKey(credentials, Id));
        var job = await PollJobAsync(client, parts[1], auth, headers, ct);
        var metadata = Metadata(job);
        var response = Response(job, parts[0]);
        if (Failed(job)) return new VideoOperationErrorResult { Error = Failure(job), ProviderMetadata = metadata, Response = response };
        if (job.Status != "succeeded" || await DownloadAsync(client, job.Id, "video/mp4", auth, headers, ct) is not { } video)
            return new VideoOperationPendingResult { ProviderMetadata = metadata, Response = response };
        return new VideoOperationCompletedResult { ProviderMetadata = metadata, Response = response,
            Videos = [new VideoOperationVideoData { Type = "base64", Data = video.Data, MediaType = video.Mime }] };
    }

    private async Task<string> UploadAsync(HttpClient client, VideoFile source, Dictionary<string, JsonElement> options,
        Action<HttpRequestHeaders> auth, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var (bytes, mime) = ReadInline(source.Data, "video/", source.MediaType ?? "video/mp4");
        var extension = mime switch { "video/webm" => "webm", "video/quicktime" => "mov", "video/x-matroska" => "mkv", "video/x-msvideo" => "avi", _ => "mp4" };
        var fileName = Option(options, "file_name", "fileName") ?? $"input.{extension}";
        var (upload, _) = await NativeHttp.PostJsonAsync(client, "uploads", new { file_name = fileName, file_size = bytes.LongLength, content_type = mime }, auth, headers, Json, ct);
        if (String(upload, "upload_id") is not { Length: > 0 } uploadId) throw new JsonException("VanceAI returned no upload ID.");
        using var transfer = clients.CreateClient(TransferClient);
        using var put = new HttpRequestMessage(HttpMethod.Put, PublicMediaDelivery.RequirePublicHttpsUri(String(upload, "put_url"))) { Content = new ByteArrayContent(bytes) };
        put.Content.Headers.ContentType = new(mime);
        using var result = await transfer.SendAsync(put, HttpCompletionOption.ResponseHeadersRead, ct);
        if (!result.IsSuccessStatusCode) throw new IOException("VanceAI video upload failed.");
        return uploadId;
    }

    private async Task<(string Data, string Mime)?> DownloadAsync(HttpClient client, string jobId, string fallback,
        Action<HttpRequestHeaders> auth, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, $"jobs/{jobId}/result");
        auth(request.Headers);
        NativeHttp.ApplyHeaders(request, headers);
        using var result = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
        if (result.StatusCode == HttpStatusCode.Accepted) return null;
        if ((int)result.StatusCode is 301 or 302 or 303 or 307 or 308)
        {
            using var transfer = clients.CreateClient(TransferClient);
            var uri = PublicMediaDelivery.RequirePublicHttpsUri(result.Headers.Location?.OriginalString);
            using var asset = await transfer.GetAsync(uri, HttpCompletionOption.ResponseHeadersRead, ct);
            if (!asset.IsSuccessStatusCode) throw new IOException("VanceAI result delivery failed.");
            return await ReadResultAsync(asset, fallback, ct);
        }
        await NativeHttp.EnsureSuccessAsync(result, request, client.DefaultRequestHeaders, ct);
        return await ReadResultAsync(result, fallback, ct);
    }

    private static async Task<(string Data, string Mime)> ReadResultAsync(HttpResponseMessage result, string fallback, CancellationToken ct)
    {
        if (result.Content.Headers.ContentLength > MaxBytes) throw new ProviderRequestException("VanceAI result exceeds the inline size limit.", 413);
        var mime = result.Content.Headers.ContentType?.MediaType?.ToLowerInvariant();
        if (mime is null or "application/octet-stream") mime = fallback;
        if (!mime.StartsWith(fallback[..(fallback.IndexOf('/') + 1)], StringComparison.Ordinal)) throw new IOException("Unexpected VanceAI result media type.");
        await using var stream = await result.Content.ReadAsStreamAsync(ct);
        var data = await BoundedBase64Async(stream, MaxBytes, ct);
        if (data.Length == 0) throw new IOException("VanceAI returned an empty result.");
        return (data, mime);
    }

    private static HttpRequestMessage JobRequest(IReadOnlyDictionary<string, string> headers, Action<HttpRequestHeaders> auth)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, "jobs");
        auth(request.Headers);
        request.Headers.Add("Idempotency-Key", Guid.NewGuid().ToString("N"));
        NativeHttp.ApplyHeaders(request, headers);
        return request;
    }

    private static async Task<Job> PollJobAsync(HttpClient client, string id, Action<HttpRequestHeaders> auth,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, $"jobs/{id}");
        auth(request.Headers);
        NativeHttp.ApplyHeaders(request, headers);
        return await ReadJobAsync(client, request, ct, id);
    }

    private static async Task<Job> ReadJobAsync(HttpClient client, HttpRequestMessage request, CancellationToken ct, string? expectedId = null)
    {
        using var upstream = await NativeHttp.SendAsync(client, request, ct);
        var raw = await NativeHttp.ReadAsync<JsonElement>(upstream, Json, ct);
        if (String(raw, "job_id") is not { } id || !ValidId(id) || expectedId is not null && id != expectedId)
            throw new JsonException("VanceAI returned an invalid job ID.");
        var safe = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, values) in upstream.Headers.Concat(upstream.Content.Headers)) safe[name] = string.Join(", ", values);
        var key = request.Headers.Authorization?.Parameter;
        return new(id, String(raw, "status")?.ToLowerInvariant() ?? "unknown", Scrub(raw, key),
            NativeHttp.SafeResponseMetadataHeaders(safe).Where(pair => pair.Value is not null && !Sensitive(pair.Value, key))
                .ToDictionary(pair => pair.Key, pair => pair.Value!, StringComparer.OrdinalIgnoreCase));
    }

    private static bool ValidId(string id) => id.Length is > 0 and <= 200 && id is not ("." or "..")
        && id.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_' or '.');
    private static bool IsTool(string tool, string type) => Tools.Value.TryGetValue(tool, out var model) && model.Type == type;
    private static string RequireTool(string tool, string type) => IsTool(tool, type) ? tool.ToLowerInvariant()
        : throw new ProviderRequestException($"Unknown VanceAI {type} tool.");
    private static bool Failed(Job job) => job.Status is "failed" or "canceled" or "cancelled";
    private static string Failure(Job job) => ProviderErrorText.FromJson(job.Raw) ?? String(job.Raw, "error") ?? $"VanceAI job ended with status '{job.Status}'.";
    private static Dictionary<string, JsonElement> Metadata(Job job) => new() { [Id] = job.Raw };
    private static HeaderResponseData Response(Job job, string tool) => new() { Timestamp = DateTime.UtcNow, ModelId = tool, Headers = job.Headers };
    private sealed record Job(string Id, string Status, JsonElement Raw, Dictionary<string, string> Headers);

    private static readonly HashSet<string> NoReservedFields = new(StringComparer.OrdinalIgnoreCase);
    private static JsonObject Config(Dictionary<string, JsonElement> options)
    {
        var config = new JsonObject();
        if (options.TryGetValue("config", out var native))
        {
            if (native.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("VanceAI config must be an object.");
            foreach (var item in native.EnumerateObject())
                if (!CredentialField(item.Name)) config[item.Name] = Copy(item.Value);
        }
        foreach (var (name, value) in options)
            if (!TransportFields.Contains(name) && !CredentialField(name)) config[name] = Copy(value);
        return config;
    }

    private static string? Option(Dictionary<string, JsonElement> options, string name, string alias)
        => (options.TryGetValue(name, out var value) || options.TryGetValue(alias, out value)) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;

    private static string? SourceUrl(string? data)
    {
        if (!Uri.TryCreate(data, UriKind.Absolute, out var uri) || uri.Scheme is not ("http" or "https")) return null;
        if (uri.IsLoopback || uri.HostNameType != UriHostNameType.Dns || !uri.IsDefaultPort || uri.UserInfo.Length != 0 || uri.Fragment.Length != 0)
            throw new ProviderRequestException("VanceAI requires a public HTTP(S) source URL.");
        return uri.AbsoluteUri; // VanceAI fetches the source; the gateway never connects to it.
    }

    private static (byte[] Bytes, string Mime) ReadInline(string? data, string category, string fallback)
    {
        if (string.IsNullOrWhiteSpace(data)) throw new ProviderRequestException("VanceAI requires inline media or a source URL.");
        var mime = fallback;
        if (data.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
        {
            var marker = data.IndexOf(";base64,", StringComparison.OrdinalIgnoreCase);
            if (marker is < 6 or > 64) throw new ProviderRequestException("Invalid base64 media data URL.");
            mime = data[5..marker];
            data = data[(marker + 8)..];
        }
        mime = mime.ToLowerInvariant();
        if (!MediaTypeHeaderValue.TryParse(mime, out var parsed) || parsed.Parameters.Count != 0
            || !mime.StartsWith(category, StringComparison.Ordinal)) throw new ProviderRequestException("Unexpected VanceAI input media type.", 415);
        if (data.Length > (MaxBytes + 2) / 3 * 4) throw new ProviderRequestException("VanceAI input is too large.", 413);
        try
        {
            var bytes = Convert.FromBase64String(data);
            if (bytes.Length is 0 or > MaxBytes) throw new ProviderRequestException("Invalid VanceAI input size.", 413);
            return (bytes, mime);
        }
        catch (FormatException) { throw new ProviderRequestException("Invalid VanceAI input base64."); }
    }

    private static string ImageFormat(string mime) => mime switch
    {
        "image/jpeg" or "image/jpg" => "jpeg", "image/png" => "png", "image/webp" => "webp",
        _ => throw new ProviderRequestException("VanceAI supports JPEG, PNG, and WebP images.", 415)
    };

    private static bool CredentialField(string name) => name.Replace("-", "").Replace("_", "").ToLowerInvariant()
        is "authorization" or "apikey" or "xapikey" or "accesstoken" or "refreshtoken" or "token" or "secret" or "clientsecret"
        or "cookie" or "setcookie" or "puturl" or "signedurl" or "presignedurl" or "uploadurl" or "downloadurl";

    private static bool Sensitive(string content, string? key) => key is { Length: > 0 }
        && (content == key || key.Length >= 8 && content.Contains(key, StringComparison.Ordinal))
        || content.Contains('?') && (content.Contains("https://", StringComparison.OrdinalIgnoreCase) || content.Contains("http://", StringComparison.OrdinalIgnoreCase));

    private static JsonElement Scrub(JsonElement value, string? key)
    {
        var node = Copy(value);
        Clean(node);
        return JsonSerializer.SerializeToElement(node, Json);

        bool SensitiveNode(JsonNode? item) => item is JsonValue text && text.TryGetValue<string>(out var content) && Sensitive(content, key);

        void Clean(JsonNode? item)
        {
            if (item is JsonObject obj)
                foreach (var (name, child) in obj.ToArray())
                    if (CredentialField(name) || SensitiveNode(child)) obj.Remove(name);
                    else Clean(child);
            else if (item is JsonArray array)
                for (var i = 0; i < array.Count; i++)
                    if (SensitiveNode(array[i])) array[i] = null;
                    else Clean(array[i]);
        }
    }
}
