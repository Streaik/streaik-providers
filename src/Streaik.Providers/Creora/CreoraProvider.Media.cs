using System.Text.Json;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class CreoraProvider
{
    private const int VideoLimit = 64 * 1024 * 1024;

    public async Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var payload = NativeOptionsPayload.Build(request, Id, Json);
        payload.Remove("stream");
        payload.Remove("partial_images");
        using var client = CreateClient();
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, "images/generations", payload, Bearer(ApiKey()), headers, Json, ct);
        var result = body.Deserialize<OpenAIImagesResponse>(Json) ?? throw new JsonException("Missing Creora image response.");
        if (result.Data is not { Count: > 0 } || result.Data.Any(image => string.IsNullOrWhiteSpace(image.B64Json)))
            throw new JsonException("Creora did not return base64 images.");
        result.OutputFormat ??= "webp";
        // Carry nested provider fields without duplicating the image bytes in every translated response.
        var metadata = body.EnumerateObject().Where(field => field.Name != "data")
            .ToDictionary(field => field.Name, field => (object)field.Value, StringComparer.Ordinal);
        metadata["data"] = Array(Property(body, "data")).Select(item => item.EnumerateObject()
            .Where(field => field.Name != "b64_json").ToDictionary(field => field.Name, field => field.Value)).ToArray();
        var safeHeaders = NativeHttp.SafeResponseMetadataHeaders(upstream);
        if (safeHeaders.Count > 0) metadata.TryAdd("headers", safeHeaders);
        (result.AdditionalProperties ??= []).TryAdd("providerMetadata", JsonSerializer.SerializeToElement(new Dictionary<string, object> { [Id] = metadata }, Json));
        return result;
    }

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => throw new ProviderRequestException("Creora does not support image editing or reference images.", 422);

    public async Task<VideoOperationStartResult> StartVideoAsync(VideoRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Image is not null || request.InputReferences?.Any() == true || request.FrameImages?.Any() == true)
            throw new ProviderRequestException("Creora does not support reference video inputs.", 422);
        var payload = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        if (request.ProviderOptions?.TryGetValue(Id, out var options) == true)
        {
            if (options.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Creora video options must be an object.");
            foreach (var option in options.EnumerateObject())
                if (!VideoReserved.Contains(option.Name)) payload[option.Name] = option.Value;
        }
        payload["model"] = JsonSerializer.SerializeToElement(request.Model);
        payload["prompt"] = JsonSerializer.SerializeToElement(request.Prompt);
        using var client = CreateClient();
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, "videos", payload, Bearer(ApiKey()), headers, Json, ct);
        var operation = String(body, "id");
        if (!ValidOperation(operation)) throw new JsonException("Missing or invalid Creora video operation ID.");
        var warnings = new List<object>();
        void Warn(object? value, string feature) { if (value is not null) warnings.Add(new { type = "unsupported", feature }); }
        Warn(request.Duration, "duration"); Warn(request.Resolution, "resolution"); Warn(request.AspectRatio, "aspectRatio");
        Warn(request.Fps, "fps"); Warn(request.N, "n"); Warn(request.Seed, "seed"); Warn(request.GenerateAudio, "generateAudio");
        return new() { Operation = operation!, Warnings = warnings, ProviderMetadata = new() { [Id] = body },
            Response = ResponseInfo(request.Model, body, upstream) };
    }

    public async Task<VideoOperationStatusResult> GetVideoAsync(string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!ValidOperation(operation)) throw new ProviderRequestException("Invalid Creora video operation ID.");
        var path = $"videos/{Uri.EscapeDataString(operation)}";
        using var client = CreateClient();
        var key = ApiKey();
        using var upstream = await SendAsync(client, HttpMethod.Get, path, null, "Bearer", key, headers, ct);
        var body = await NativeHttp.ReadAsync<JsonElement>(upstream, Json, ct);
        var info = ResponseInfo(String(body, "model") ?? "creora-video", body, upstream.Headers.Concat(upstream.Content.Headers)
            .ToDictionary(pair => pair.Key, pair => (string?)string.Join(", ", pair.Value), StringComparer.OrdinalIgnoreCase));
        var metadata = new Dictionary<string, JsonElement> { [Id] = body };
        switch (String(body, "status"))
        {
            case "queued" or "in_progress":
                return new VideoOperationPendingResult { Response = info, ProviderMetadata = metadata };
            case "failed":
                return new VideoOperationErrorResult { Error = "Creora video generation failed.", Response = info, ProviderMetadata = metadata };
            case "completed":
                using (var content = await SendAsync(client, HttpMethod.Get, path + "/content", null, "Bearer", key, headers, ct))
                {
                    var mime = content.Content.Headers.ContentType?.MediaType ?? "video/mp4";
                    if (!mime.StartsWith("video/", StringComparison.OrdinalIgnoreCase) && mime != "application/octet-stream")
                        throw new IOException("Creora returned an unexpected video content type.");
                    if (content.Content.Headers.ContentLength > VideoLimit) throw new IOException("Creora video exceeds the size limit.");
                    await using var input = await content.Content.ReadAsStreamAsync(ct);
                    var data = await BoundedBase64Async(input, VideoLimit, ct);
                    return new VideoOperationCompletedResult { Response = info, ProviderMetadata = metadata,
                        Videos = [new() { Type = "base64", MediaType = mime == "application/octet-stream" ? "video/mp4" : mime, Data = data }] };
                }
            default:
                throw new JsonException("Unknown Creora video status.");
        }
    }

    private static readonly HashSet<string> VideoReserved = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "prompt", "stream", "store", "background", "id", "image", "inputReferences", "input_references", "frameImages", "frame_images",
        "headers", "authorization", "api_key", "apiKey", "x-api-key"
    };

    private static bool ValidOperation(string? operation) => !string.IsNullOrWhiteSpace(operation) && operation is not ("." or "..")
        && !operation.Any(c => char.IsControl(c) || c is '/' or '\\');

    private static HeaderResponseData ResponseInfo(string model, JsonElement body, Dictionary<string, string?> upstream)
    {
        var timestamp = Property(body, "created");
        if (timestamp.ValueKind != JsonValueKind.Number) timestamp = Property(body, "created_at");
        return new() { ModelId = model,
            Timestamp = timestamp.ValueKind == JsonValueKind.Number && timestamp.TryGetInt64(out var epoch) && epoch is >= 0 and <= 253402300799
                ? DateTimeOffset.FromUnixTimeSeconds(epoch).UtcDateTime : DateTime.UtcNow,
            Headers = NativeHttp.SafeResponseMetadataHeaders(upstream).Where(pair => pair.Value is not null)
                .ToDictionary(pair => pair.Key, pair => pair.Value!, StringComparer.OrdinalIgnoreCase) };
    }
}
