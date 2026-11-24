using System.Buffers.Text;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.AtlasCloud;

internal sealed partial class AtlasCloudProvider
{
    private const string MediaPath = "api/v1/model/";
    private const int ImageLimit = 32 * 1024 * 1024;
    private const int VideoLimit = 64 * 1024 * 1024;

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => ImagesAsync(request.Model, request.Prompt, request.Size, request.N, request.AdditionalProperties, [], headers, ct);

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        // As in AI Happey, the edit entry point uses the text-to-image backend; it does not apply the input files or mask.
        List<object> warnings = [Warning("files", "AtlasCloud generates from the prompt only; input files are ignored.")];
        if (request.Mask is not null || request.MaskFile is not null) warnings.Add(Warning("mask"));
        return ImagesAsync(request.Model, request.Prompt, request.Size, request.N, request.AdditionalProperties, warnings, headers, ct);
    }

    private async Task<OpenAIImagesResponse> ImagesAsync(string model, string prompt, string? size, int? n,
        Dictionary<string, JsonElement>? options, List<object> warnings, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(prompt)) throw new ProviderRequestException("An image prompt is required.");
        if (!string.IsNullOrWhiteSpace(size)) warnings.Add(Warning("size", "Use the native aspect_ratio option instead."));
        if (n is > 1) warnings.Add(Warning("n", "AtlasCloud determines the number of outputs."));
        var payload = ImageOptions(options);
        payload["model"] = model;
        payload["prompt"] = prompt;
        payload["enable_base64_output"] = true;

        using var client = CreateClient();
        var auth = Bearer(ApiKey());
        var (submit, _) = await NativeHttp.PostJsonAsync(client, MediaPath + "generateImage", payload, auth, headers, Json, ct);
        var job = submit;
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromMinutes(5));
        try
        {
            if (NeedsPoll(job))
            {
                var id = JobId(submit);
                job = await PollAsync(client, id, auth, deadline.Token);
                while (NeedsPoll(job))
                {
                    await Task.Delay(TimeSpan.FromSeconds(2), deadline.Token);
                    job = await PollAsync(client, id, auth, deadline.Token);
                }
            }
        }
        catch (OperationCanceledException ex) when (!ct.IsCancellationRequested)
        { throw new IOException("AtlasCloud image generation timed out.", ex); }
        if (Failed(job)) throw new IOException("AtlasCloud image generation failed.");

        var media = await ReadOutputsAsync(job, "image/png", ImageLimit, ct);
        if (media.Count == 0) throw new IOException("AtlasCloud returned no images.");
        return new()
        {
            Created = Timestamp(job).ToUnixTimeSeconds(),
            Data = media.Select(image => new OpenAIImageData { B64Json = image.Data }).ToList(),
            OutputFormat = media[0].Mime[6..] is "svg+xml" ? "svg" : media[0].Mime[6..],
            AdditionalProperties = new()
            {
                ["providerMetadata"] = JsonSerializer.SerializeToElement(new Dictionary<string, object>
                    { [Id] = new { submit = WithoutOutputs(submit), result = WithoutOutputs(job) } }, Json),
                ["warnings"] = JsonSerializer.SerializeToElement(warnings, Json)
            }
        };
    }

    public async Task<VideoOperationStartResult> StartVideoAsync(VideoRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var payload = new JsonObject(new JsonNodeOptions { PropertyNameCaseInsensitive = true });
        if (request.ProviderOptions?.TryGetValue(Id, out var options) == true)
        {
            if (options.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("AtlasCloud options must be a JSON object.");
            AddOptions(payload, options.EnumerateObject().Select(p => KeyValuePair.Create(p.Name, p.Value)), "model", "prompt", "image");
        }
        payload["model"] = request.Model;
        if (!string.IsNullOrWhiteSpace(request.Prompt)) payload["prompt"] = request.Prompt;
        if (request.Duration is { } duration) payload["duration"] = duration;
        if (!string.IsNullOrWhiteSpace(request.Resolution)) payload["resolution"] = request.Resolution;
        if (!string.IsNullOrWhiteSpace(request.AspectRatio)) payload["aspect_ratio"] = request.AspectRatio;
        if (request.Seed is { } seed) payload["seed"] = seed;
        if (request.Image is { } image) payload["image"] = VideoImage(image);
        List<object> warnings = [];
        if (request.Fps is not null) warnings.Add(Warning("fps"));
        if (request.N is > 1) warnings.Add(Warning("n", "AtlasCloud determines the number of outputs."));
        if (request.FrameImages?.Any() == true) warnings.Add(Warning("frameImages"));
        if (request.InputReferences?.Any() == true) warnings.Add(Warning("inputReferences"));
        if (request.GenerateAudio is { } generateAudio) payload["generate_audio"] = generateAudio;

        using var client = CreateClient();
        var (job, responseHeaders) = await NativeHttp.PostJsonAsync(client, MediaPath + "generateVideo", payload, Bearer(ApiKey()), headers, Json, ct);
        return new()
        {
            Operation = JobId(job), Warnings = warnings, ProviderMetadata = JobMetadata(job),
            Response = new() { ModelId = request.Model, Timestamp = Timestamp(job).UtcDateTime,
                Headers = NativeHttp.SafeResponseMetadataHeaders(responseHeaders).Where(p => p.Value is not null)
                    .ToDictionary(p => p.Key, p => p.Value!, StringComparer.OrdinalIgnoreCase) }
        };
    }

    public async Task<VideoOperationStatusResult> GetVideoAsync(string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        var job = await PollAsync(client, operation, Bearer(ApiKey()), ct);
        var response = new HeaderResponseData { ModelId = String(JobData(job), "model") ?? Id, Timestamp = Timestamp(job).UtcDateTime };
        var metadata = JobMetadata(job);
        if (Failed(job))
            return new VideoOperationErrorResult { Error = "AtlasCloud video generation failed.", Response = response, ProviderMetadata = metadata };
        if (!Completed(job) && !Outputs(job).Any())
            return new VideoOperationPendingResult { Response = response, ProviderMetadata = metadata };
        var videos = await ReadOutputsAsync(job, "video/mp4", VideoLimit, ct);
        if (videos.Count == 0)
            return new VideoOperationErrorResult { Error = "AtlasCloud returned no videos.", Response = response, ProviderMetadata = metadata };
        return new VideoOperationCompletedResult { Videos = videos.Select(video => new VideoOperationVideoData
            { Type = "base64", MediaType = video.Mime, Data = video.Data }).ToList(), Response = response, ProviderMetadata = metadata };
    }

    private static JsonObject ImageOptions(Dictionary<string, JsonElement>? options)
    {
        var payload = new JsonObject(new JsonNodeOptions { PropertyNameCaseInsensitive = true });
        if (options is null) return payload;
        var reserved = new[] { "model", "prompt", "stream", "partial_images", "response_format", "enable_base64_output", "size", "n",
            "file", "files", "images", "mask", "provider_options", "providerOptions", Id };
        AddOptions(payload, options, reserved);
        foreach (var key in new[] { "provider_options", "providerOptions" })
            if (Property(options.GetValueOrDefault(key), Id) is { ValueKind: JsonValueKind.Object } nested)
                AddOptions(payload, nested.EnumerateObject().Select(p => KeyValuePair.Create(p.Name, p.Value)), reserved);
        if (options.GetValueOrDefault(Id) is { ValueKind: JsonValueKind.Object } direct)
            AddOptions(payload, direct.EnumerateObject().Select(p => KeyValuePair.Create(p.Name, p.Value)), reserved);
        return payload;
    }

    private static void AddOptions(JsonObject payload, IEnumerable<KeyValuePair<string, JsonElement>> options, params string[] reserved)
    {
        foreach (var (name, value) in options)
            if (!reserved.Contains(name, StringComparer.OrdinalIgnoreCase)) payload[name] = Copy(value);
    }

    private static string VideoImage(VideoFile image)
    {
        var reference = image is VideoFileUrl urlFile ? urlFile.Url : image.Data;
        if (string.IsNullOrWhiteSpace(reference)) throw new ProviderRequestException("An image reference must not be empty.");
        var data = reference.Trim();
        // Input references are forwarded as data; the gateway never downloads them.
        if (Uri.TryCreate(data, UriKind.Absolute, out var url) && url.Scheme is "http" or "https" && url.UserInfo.Length == 0) return data;
        if (image is VideoFileUrl) throw new ProviderRequestException("An image URL must use HTTP or HTTPS.");
        if (data.StartsWith("data:image/", StringComparison.OrdinalIgnoreCase)) return data;
        return $"data:{image.MediaType ?? "image/jpeg"};base64,{data}";
    }

    private static Task<JsonElement> PollAsync(HttpClient client, string id, Action<HttpRequestHeaders> auth, CancellationToken ct)
    {
        if (id.Length is 0 or > 256 || !id.All(ch => char.IsAsciiLetterOrDigit(ch) || ch is '-' or '_'))
            throw new IOException("AtlasCloud returned an invalid job ID.");
        return NativeHttp.GetAsync<JsonElement>(client, MediaPath + "prediction/" + id, auth, ct);
    }

    private static JsonElement JobData(JsonElement job) => Property(job, "data") is { ValueKind: JsonValueKind.Object } data ? data : job;
    private static string JobId(JsonElement job) => String(JobData(job), "id") ?? throw new IOException("AtlasCloud did not return a job ID.");
    private static bool Failed(JsonElement job) => String(JobData(job), "status")?.ToLowerInvariant() is "failed" or "error" or "canceled" or "cancelled";
    private static bool Completed(JsonElement job) => String(JobData(job), "status")?.ToLowerInvariant() is "completed" or "succeeded";
    private static bool NeedsPoll(JsonElement job) => !Failed(job) && !Completed(job) && !Outputs(job).Any();
    private static IEnumerable<string> Outputs(JsonElement job) => Array(Property(JobData(job), "outputs"))
        .Where(x => x.ValueKind == JsonValueKind.String).Select(x => x.GetString()!).Where(x => !string.IsNullOrWhiteSpace(x));
    private static DateTimeOffset Timestamp(JsonElement job) => Property(JobData(job), "created_at") is { ValueKind: JsonValueKind.String } date
        && date.TryGetDateTimeOffset(out var created) ? created : DateTimeOffset.UtcNow;

    private static object Warning(string feature, string? details = null) => new { type = "unsupported", feature, details };
    private static Dictionary<string, JsonElement> JobMetadata(JsonElement job) => new() { [Id] = WithoutOutputs(job) };

    private static JsonElement WithoutOutputs(JsonElement job)
    {
        // The existing media field already carries the output; do not duplicate base64 or expose signed URLs.
        var copy = JsonObject.Create(job) ?? throw new IOException("AtlasCloud returned invalid job metadata.");
        copy.Remove("outputs");
        if (copy["data"] is JsonObject data) data.Remove("outputs");
        return JsonSerializer.SerializeToElement(copy, Json);
    }

    private async Task<List<(string Data, string Mime)>> ReadOutputsAsync(JsonElement job, string fallback, int limit, CancellationToken ct)
    {
        var media = new List<(string Data, string Mime)>();
        var remaining = limit;
        foreach (var value in Outputs(job))
        {
            ct.ThrowIfCancellationRequested();
            var trimmed = value.Trim();
            var item = !trimmed.StartsWith("data:", StringComparison.OrdinalIgnoreCase) && Uri.TryCreate(trimmed, UriKind.Absolute, out var url)
                ? await DownloadAsync(url, fallback, remaining, ct) : Inline(trimmed, fallback);
            if (!Base64.IsValid(item.Data, out var size) || size == 0) throw new IOException("AtlasCloud returned invalid base64 media.");
            if (size > remaining) throw new IOException("AtlasCloud media exceeds the gateway's inline size limit.");
            remaining -= size;
            media.Add(item);
        }
        return media;
    }

    private async Task<(string Data, string Mime)> DownloadAsync(Uri url, string fallback, int limit, CancellationToken ct)
    {
        var host = url.IdnHost;
        if (url.Scheme != Uri.UriSchemeHttps || !url.IsDefaultPort || url.UserInfo.Length != 0 || url.Fragment.Length != 0
            || !(host == "atlascloud.ai" || host.EndsWith(".atlascloud.ai", StringComparison.Ordinal)))
            throw new IOException("AtlasCloud returned an untrusted asset URL.");
        using var client = CreateClient(AssetClient);
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        using var response = await NativeHttp.SendAsync(client, request, ct);
        var mime = response.Content.Headers.ContentType?.MediaType;
        if (!MediaTypeMatches(mime, fallback)) mime = Path.GetExtension(url.AbsolutePath).ToLowerInvariant() switch
        {
            ".webp" when fallback == "image/png" => "image/webp",
            ".jpg" or ".jpeg" when fallback == "image/png" => "image/jpeg",
            ".webm" when fallback == "video/mp4" => "video/webm",
            ".mov" when fallback == "video/mp4" => "video/quicktime",
            _ => fallback
        };
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        return (await BoundedBase64Async(stream, limit, ct), mime!);
    }

    private static (string Data, string Mime) Inline(string value, string fallback)
    {
        if (!value.StartsWith("data:", StringComparison.OrdinalIgnoreCase)) return (value, fallback);
        var comma = value.IndexOf(',');
        if (comma < 6 || value[5..comma].Split(';') is not [var mime, var encoding]
            || !encoding.Equals("base64", StringComparison.OrdinalIgnoreCase) || !MediaTypeMatches(mime, fallback))
            throw new IOException("AtlasCloud returned an invalid media data URL.");
        return (value[(comma + 1)..], mime.ToLowerInvariant());
    }

    private static bool MediaTypeMatches(string? mime, string fallback) => mime is not null
        && mime.StartsWith(fallback[..(fallback.IndexOf('/') + 1)], StringComparison.OrdinalIgnoreCase)
        && MediaTypeHeaderValue.TryParse(mime, out _);
}
