using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class DecartRegistration
{
    public static IServiceCollection AddDecart(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, "decart", "https://api.decart.ai/v1/");
        return services.AddProvider<DecartProvider>("decart");
    }
}

internal sealed class DecartProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IImageModelRoutingProvider, IVideoProvider
{
    private const string Id = "decart";
    private const int ImageLimit = 25 * 1024 * 1024;
    private const int VideoLimit = 64 * 1024 * 1024;
    private static readonly (string Id, string Name, string Type)[] Models =
    [
        ("lucy-image-2", "Lucy Image 2", "image"), ("lucy-image-latest", "Lucy Image Latest", "image"),
        ("lucy-2.5", "Lucy 2.5", "video"), ("lucy-2.1", "Lucy 2.1", "video"),
        ("lucy-vton-3", "Lucy VTON 3", "video"), ("lucy-restyle-2", "Lucy Restyle 2", "video")
    ];

    public IAsyncEnumerable<Model> ListModelsAsync(CancellationToken ct)
        => (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id)) ? [] : Models
            .Select(model => new Model { Id = model.Id, Name = model.Name, Type = model.Type, OwnedBy = Id })).ToAsyncEnumerable();

    public Task<bool> IsImageModelAsync(string model, CancellationToken ct) => Task.FromResult(IsModel(model, "image"));

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => throw new ProviderRequestException("Decart image models require an image edit input.", 422);

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!IsModel(request.Model, "image")) throw new ProviderRequestException("Unsupported Decart image model.");
        if (string.IsNullOrWhiteSpace(request.Prompt)) throw new ProviderRequestException("An image edit prompt is required.");
        var extensions = request.AdditionalProperties;
        if (StringOption(extensions, "response_format") is not (null or "b64_json")) throw new ProviderRequestException("Decart returns inline images only.");
        using var form = new MultipartFormDataContent();
        if (request.ImageFiles?.FirstOrDefault() is { } file)
        {
            if (file.Length is <= 0 or > ImageLimit || !file.ContentType.StartsWith("image/", StringComparison.OrdinalIgnoreCase))
                throw new ProviderRequestException("A nonempty image upload within the inline size limit is required.", 422);
            AddFile(form, "data", file);
        }
        else if (request.Images?.FirstOrDefault() is { } image) AddDataImage(form, "data", image.ImageUrl);
        else throw new ProviderRequestException("Decart requires an image to edit.", 422);

        var fields = RawFields(extensions, "model", "prompt", "data", "images", "image", "mask", "size", "aspect_ratio", "n", "seed",
            "stream", "partial_images", "response_format", "output_format", "resolution");
        fields["prompt"] = request.Prompt;
        fields["resolution"] = Resolution(request.Size) ?? Resolution(StringOption(extensions, "resolution"))
            ?? Resolution(StringOption(extensions, "aspect_ratio")) ?? "720p";
        if (extensions?.GetValueOrDefault("seed") is { ValueKind: JsonValueKind.Number } seed) fields["seed"] = Copy(seed);
        AddFields(form, fields);
        using var client = clients.CreateClient(Id);
        using var response = await SendAsync(client, HttpMethod.Post, $"generate/{request.Model.ToLowerInvariant()}", form, headers, ct);
        var format = response.Content.Headers.ContentType?.MediaType switch
        {
            "image/png" => "png", "image/webp" => "webp", "image/jpeg" or "application/octet-stream" or null => "jpeg",
            _ => throw new IOException("Decart did not return a supported image type.")
        };
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        var warnings = new List<object>();
        Warn(warnings, "n", request.N is > 1);
        Warn(warnings, "mask", request.Mask is not null || request.MaskFile is not null);
        Warn(warnings, "images", (request.Images?.Length ?? 0) + (request.ImageFiles?.Length ?? 0) > 1);
        return new() { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), OutputFormat = format,
            Data = [new() { B64Json = await BoundedBase64Async(stream, ImageLimit, ct) }],
            AdditionalProperties = warnings.Count == 0 ? null : new() { ["warnings"] = JsonSerializer.SerializeToElement(warnings, Json) } };
    }

    public async Task<VideoOperationStartResult> StartVideoAsync(VideoRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!IsModel(request.Model, "video")) throw new ProviderRequestException("Unsupported Decart video model.");
        var inputs = new[] { request.Image }.Concat(request.InputReferences ?? []).OfType<VideoFile>().ToArray();
        var video = inputs.FirstOrDefault(x => x.MediaType?.StartsWith("video/", StringComparison.OrdinalIgnoreCase) == true)
            ?? throw new ProviderRequestException("Decart requires an inline video in image or inputReferences.", 422);
        using var form = new MultipartFormDataContent();
        AddInline(form, "data", video, VideoLimit, "input-video");
        if (inputs.FirstOrDefault(x => x.MediaType?.StartsWith("image/", StringComparison.OrdinalIgnoreCase) == true) is { } reference)
            AddInline(form, "reference_image", reference, ImageLimit, "reference-image");
        var options = request.ProviderOptions?.GetValueOrDefault(Id);
        if (options is { ValueKind: not (JsonValueKind.Object or JsonValueKind.Null or JsonValueKind.Undefined) })
            throw new ProviderRequestException("Decart provider options must be an object.");
        var fields = RawFields(options is { ValueKind: JsonValueKind.Object } obj
            ? obj.EnumerateObject().ToDictionary(x => x.Name, x => x.Value, StringComparer.Ordinal) : null,
            "model", "prompt", "data", "video", "image", "reference_image", "inputReferences", "frameImages", "generateAudio",
            "n", "fps", "duration", "resolution", "aspectRatio", "aspect_ratio", "seed", "stream");
        if (!string.IsNullOrWhiteSpace(request.Prompt)) fields["prompt"] = request.Prompt;
        var resolution = Resolution(request.Resolution);
        fields["resolution"] = resolution ?? Resolution(request.AspectRatio) ?? "720p";
        fields["seed"] = request.Seed;
        AddFields(form, fields);
        var warnings = new List<object>();
        Warn(warnings, "n", request.N is > 1);
        Warn(warnings, "fps", request.Fps.HasValue);
        Warn(warnings, "duration", request.Duration.HasValue);
        Warn(warnings, "resolution", request.Resolution is not null && resolution is null);
        Warn(warnings, "frameImages", request.FrameImages?.Any() == true);
        Warn(warnings, "generateAudio", request.GenerateAudio.HasValue);
        using var client = clients.CreateClient(Id);
        using var response = await SendAsync(client, HttpMethod.Post, $"jobs/{request.Model.ToLowerInvariant()}", form, headers, ct);
        var job = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        var id = String(job, "job_id");
        if (!ValidJobId(id)) throw new JsonException("Decart did not return a valid job ID.");
        return new() { Operation = id!, Warnings = warnings, ProviderMetadata = JobMetadata(job), Response = new() { Timestamp = DateTime.UtcNow } };
    }

    public async Task<VideoOperationStatusResult> GetVideoAsync(string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!ValidJobId(operation)) throw new ProviderRequestException("Invalid Decart job reference.");
        using var client = clients.CreateClient(Id);
        var path = $"jobs/{Uri.EscapeDataString(operation)}";
        using var response = await SendAsync(client, HttpMethod.Get, path, null, headers, ct);
        var job = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        var status = String(job, "status")?.ToLowerInvariant() ?? throw new JsonException("Decart did not return a job status.");
        VideoOperationStatusResult result;
        if (status == "completed")
        {
            using var content = await SendAsync(client, HttpMethod.Get, path + "/content", null, headers, ct);
            await using var stream = await content.Content.ReadAsStreamAsync(ct);
            var mime = content.Content.Headers.ContentType?.MediaType;
            if (mime is null or "application/octet-stream") mime = "video/mp4";
            if (!mime.StartsWith("video/", StringComparison.OrdinalIgnoreCase)) throw new IOException("Decart did not return video content.");
            result = new VideoOperationCompletedResult { Videos = [new() { Type = "base64", MediaType = mime,
                Data = await BoundedBase64Async(stream, VideoLimit, ct) }] };
        }
        else if (status is "failed" or "canceled" or "cancelled")
            result = new VideoOperationErrorResult { Error = $"Decart video job {status}." };
        else result = new VideoOperationPendingResult();
        result.ProviderMetadata = JobMetadata(job);
        result.Response = new() { Timestamp = DateTime.UtcNow };
        return result;
    }

    private async Task<HttpResponseMessage> SendAsync(HttpClient client, HttpMethod method, string path, HttpContent? content,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var request = new HttpRequestMessage(method, path) { Content = content };
        request.Headers.Add("X-API-KEY", RequiredKey(credentials, Id));
        NativeHttp.ApplyHeaders(request, headers);
        return await NativeHttp.SendAsync(client, request, ct);
    }

    private static void AddInline(MultipartFormDataContent form, string field, VideoFile input, int limit, string fileName)
    {
        var data = input.Data;
        if (input is VideoFileUrl || string.IsNullOrEmpty(data)) throw new ProviderRequestException("Decart requires inline media; URLs are not fetched.", 422);
        if (data.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
        {
            var comma = data.IndexOf(',');
            if (comma < 0 || !data.AsSpan(0, comma).Equals($"data:{input.MediaType};base64", StringComparison.OrdinalIgnoreCase))
                throw new ProviderRequestException("Invalid media data URL.");
            data = data[(comma + 1)..];
        }
        if (data.Length > ((long)limit + 2) / 3 * 4) throw new ProviderRequestException("Media exceeds the inline size limit.", 413);
        if (!System.Buffers.Text.Base64.IsValid(data, out var length) || length == 0) throw new ProviderRequestException("Invalid base64 media.");
        if (length > limit) throw new ProviderRequestException("Media exceeds the inline size limit.", 413);
        AddFile(form, field, new MemoryStream(Convert.FromBase64String(data), writable: false), input.MediaType!, fileName);
    }

    private static bool IsModel(string id, string type) => Models.Any(x => x.Type == type && x.Id.Equals(id, StringComparison.OrdinalIgnoreCase));
    private static bool ValidJobId(string? value) => value is { Length: > 0 and <= 256 } && value.All(ch => char.IsAsciiLetterOrDigit(ch) || ch is '-' or '_');
    private static string? Resolution(string? size) => size?.Trim().ToLowerInvariant().Replace(':', 'x') switch
    {
        "480p" or "480x832" or "832x480" or "15x26" or "26x15" => "480p",
        "720p" or "720x1280" or "1280x720" or "9x16" or "16x9" => "720p", _ => null
    };
    private static string? StringOption(IReadOnlyDictionary<string, JsonElement>? options, string name)
        => options?.GetValueOrDefault(name) is { ValueKind: JsonValueKind.String } value ? value.GetString() : null;

    private static Dictionary<string, JsonElement> JobMetadata(JsonElement job) => new() { [Id] = job };
    private static void Warn(List<object> warnings, string feature, bool unsupported)
    {
        if (unsupported) warnings.Add(new { type = "unsupported", feature });
    }

    private static JsonObject RawFields(IReadOnlyDictionary<string, JsonElement>? values, params string[] reserved)
    {
        var fields = new JsonObject();
        var blocked = reserved.Concat(["authorization", "api_key", "api-key", "x-api-key", "headers", "endpoint", "url"]).ToHashSet(StringComparer.OrdinalIgnoreCase);
        foreach (var (key, value) in values ?? new Dictionary<string, JsonElement>())
            if (!blocked.Contains(key)) fields[key] = Copy(value);
        return fields;
    }
}
