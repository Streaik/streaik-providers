using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.ImageRouter;

public static class ImageRouterRegistration
{
    public static IServiceCollection AddImageRouter(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, ImageRouterProvider.Id, "https://api.imagerouter.io/");
        return services.AddProvider<ImageRouterProvider>(ImageRouterProvider.Id);
    }
}

public sealed class ImageRouterProvider(IHttpClientFactory clients, IProviderCredentials credentials) : IImageProvider, IVideoProvider, IModelListingProvider
{
    public const string Id = "imagerouter";

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        var root = await NativeHttp.GetAsync<JsonElement>(client, "v3/models", _ => { }, ct);
        foreach (var item in Array(root))
        {
            var id = String(item, "id");
            var output = Array(Property(Property(item, "architecture"), "output_modalities")).Select(x => x.GetString()).ToArray();
            var type = output.Contains("video") ? "video" : output.Contains("image") ? "image" : null;
            if (id is { Length: > 0 } && type is not null)
                yield return new Model { Id = id, Name = String(item, "name") ?? id, Type = type };
        }
    }

    public async Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Stream == true && (request.PartialImages is > 0 || request.N is > 1))
            throw new ProviderRequestException("ImageRouter streaming supports one final image, not partial images.", 422);
        var fields = ToObject(request);
        fields.Remove("stream"); fields.Remove("partial_images");
        fields["response_format"] = request.Stream == true ? "b64_json" : request.ResponseFormat ?? "b64_json";
        return await ImageAsync("v1/openai/images/generations", JsonContent.Create(fields, options: Json), headers, ct);
    }

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Stream == true && (request.PartialImages is > 0 || request.N is > 1))
            throw new ProviderRequestException("ImageRouter streaming supports one final image, not partial images.", 422);
        var form = ImageMultipart(request, inlineReferences: true, "image[]", "mask[]");
        form.Add(new StringContent("b64_json"), "response_format");
        return ImageAsync("v1/openai/images/edits", form, headers, ct);
    }

    private async Task<OpenAIImagesResponse> ImageAsync(string path, HttpContent content, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var body = content;
        var key = RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromMinutes(2));
        var root = await JsonAsync<JsonElement>(client, HttpMethod.Post, path, body, "Bearer", key, headers, timeout.Token);
        while (!Failed(root) && !HasData(root) && PollReference(root) is { } poll)
        {
            await Task.Delay(TimeSpan.FromSeconds(2), timeout.Token);
            root = await JsonAsync<JsonElement>(client, HttpMethod.Get, PollPath(client, poll), null, "Bearer", key, headers, timeout.Token);
        }
        if (Failed(root) || !HasData(root)) throw new HttpRequestException("ImageRouter did not return an image.");
        var result = root.Deserialize<OpenAIImagesResponse>(Json) ?? throw new JsonException();
        result.Created = result.Created == 0 ? DateTimeOffset.UtcNow.ToUnixTimeSeconds() : result.Created;
        return result;
    }

    public async Task<VideoOperationStartResult> StartVideoAsync(VideoRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.InputReferences?.Any() == true || request.FrameImages?.Any() == true)
            throw new ProviderRequestException("ImageRouter video accepts one input image, not multi-frame references.", 422);
        var fields = new JsonObject();
        if (request.ProviderOptions?.TryGetValue(Id, out var options) == true && options.ValueKind == JsonValueKind.Object)
            foreach (var item in options.EnumerateObject()) fields[item.Name] = Copy(item.Value);
        fields["model"] = request.Model;
        fields["prompt"] = request.Prompt;
        if (request.Resolution is not null) fields["size"] = request.Resolution;
        if (request.AspectRatio is not null) fields["aspect_ratio"] = request.AspectRatio;
        if (request.Duration is not null) fields["seconds"] = request.Duration;
        if (request.GenerateAudio is not null) fields["generate_audio"] = request.GenerateAudio;
        if (request.Seed is not null) fields["seed"] = request.Seed;
        if (request.Fps is not null) fields["fps"] = request.Fps;
        if (request.N is not null) fields["n"] = request.N;
        using var client = clients.CreateClient(Id);
        using var payload = VideoPayload(request.Image, fields);
        var root = await JsonAsync<JsonElement>(client, HttpMethod.Post, "v1/openai/videos/generations", payload,
            "Bearer", RequiredKey(credentials, Id), headers, ct);
        if (Failed(root) || PollReference(root) is not { } poll) throw new HttpRequestException("ImageRouter did not return a video operation.");
        return new() { Operation = PollPath(client, poll), Response = new() { Timestamp = DateTime.UtcNow } };
    }

    public async Task<VideoOperationStatusResult> GetVideoAsync(string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        var root = await JsonAsync<JsonElement>(client, HttpMethod.Get, PollPath(client, operation), null,
            "Bearer", RequiredKey(credentials, Id), headers, ct);
        VideoOperationStatusResult result;
        if (Failed(root)) result = new VideoOperationErrorResult { Error = "ImageRouter video generation failed." };
        else if (HasData(root))
        {
            var videos = new List<VideoOperationVideoData>();
            foreach (var item in Array(Property(root, "data")))
            {
                if (String(item, "b64_json") is { Length: > 0 } b64) videos.Add(new() { Type = "base64", Data = b64, MediaType = "video/mp4" });
                else if (Uri.TryCreate(String(item, "url"), UriKind.Absolute, out var url) && url.Scheme == Uri.UriSchemeHttps && url.UserInfo == "")
                    videos.Add(new() { Type = "url", Url = url.AbsoluteUri, MediaType = "video/mp4" });
            }
            result = videos.Count > 0 ? new VideoOperationCompletedResult { Videos = videos }
                : new VideoOperationErrorResult { Error = "ImageRouter returned no playable video." };
        }
        else result = new VideoOperationPendingResult();
        result.Response = new() { Timestamp = DateTime.UtcNow };
        var cost = Property(root, "cost");
        if (cost.ValueKind == JsonValueKind.Number) result.ProviderMetadata = new() { [Id] = JsonSerializer.SerializeToElement(new { cost }) };
        return result;
    }

    private static HttpContent VideoPayload(VideoFile? image, JsonObject fields)
    {
        if (image is null) return JsonContent.Create(fields, options: Json);
        var form = new MultipartFormDataContent();
        try
        {
            AddFields(form, fields);
            AddDataImage(form, "image", image is VideoFileUrl url ? url.Url : $"data:{image.MediaType};base64,{image.Data}");
            return form;
        }
        catch { form.Dispose(); throw; }
    }

    private static string? PollReference(JsonElement root) => String(root, "fetch_result") ?? String(root, "result_url");
    private static bool Failed(JsonElement root) => Property(root, "error").ValueKind is not (JsonValueKind.Undefined or JsonValueKind.Null)
        || String(root, "status")?.ToLowerInvariant() is "failed" or "error" or "cancelled" or "canceled";
    private static bool HasData(JsonElement root) => Property(root, "data") is { ValueKind: JsonValueKind.Array } data && data.GetArrayLength() > 0;

    private static string PollPath(HttpClient client, string reference)
    {
        var origin = client.BaseAddress!;
        if (reference.Length > 2048 || !Uri.TryCreate(origin, reference, out var target) || target.Scheme != "https"
            || !target.IdnHost.Equals(origin.IdnHost, StringComparison.OrdinalIgnoreCase) || target.Port != origin.Port
            || target.UserInfo.Length != 0 || target.Fragment.Length != 0)
            throw new HttpRequestException("ImageRouter returned an invalid poll URL.");
        return target.AbsoluteUri;
    }
}
