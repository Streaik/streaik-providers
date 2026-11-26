using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class DeAPIProviderRegistration
{
    public static IServiceCollection AddDeAPI(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, DeAPIProvider.Id, "https://api.deapi.ai/");
        ProviderServiceCollectionExtensions.AddClient(services, DeAPIProvider.Assets, "https://api.deapi.ai/");
        return services.AddProvider<DeAPIProvider>(DeAPIProvider.Id);
    }
}

internal sealed partial class DeAPIProvider(IHttpClientFactory clients, IProviderCredentials credentials, IConfiguration configuration)
    : IModelListingProvider, IImageProvider, IStreamingImagesProvider, ISpeechProvider, ITranscriptionProvider, IVideoProvider, IEmbeddingMetadataProvider
{
    internal const string Id = "deapi";
    internal const string Assets = "deapi-assets";
    private const string V2 = "api/v2/";
    private string Key() => RequiredKey(credentials, Id);

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (credentials.GetApiKey(Id) is not { } key || string.IsNullOrWhiteSpace(key)) yield break;
        using var client = clients.CreateClient(Id);
        var lastPage = 1;
        for (var page = 1; page <= lastPage; page++)
        {
            var body = await NativeHttp.GetAsync<JsonElement>(client, $"{V2}models?page={page}", h => h.Authorization = new("Bearer", key), ct);
            if (Property(Property(body, "meta"), "last_page") is { ValueKind: JsonValueKind.Number } last && last.TryGetInt32(out var count))
                lastPage = Math.Clamp(count, 1, 1000);
            foreach (var item in Array(Property(body, "data")))
                if (String(item, "slug") is { Length: > 0 } slug)
                    yield return new Model { Id = slug, Name = String(item, "name") ?? slug, Description = String(item, "description"),
                        OwnedBy = "deapi.ai", Type = ModelType(item) };
        }
    }

    private static string ModelType(JsonElement item)
    {
        var types = Property(item, "inference_types");
        return (String(item, "inference_type") ?? (types.ValueKind == JsonValueKind.Object
            ? types.EnumerateObject().Select(p => p.Name).FirstOrDefault()
            : Array(types).Where(p => p.ValueKind == JsonValueKind.String).Select(p => p.GetString()).FirstOrDefault())) switch
        {
            "txt2img" or "img2img" or "img-rmbg" or "img-upscale" => "image",
            "txt2audio" or "txt2music" => "speech",
            "video2text" or "video_file2text" or "audio2text" or "audio_file2text" or "img2txt" => "transcription",
            "txt2video" or "img2video" or "audio2video" or "vid-upscale" or "video-replace" => "video",
            "txt2embedding" or "embedding" => "embedding",
            _ => "language"
        };
    }

    public async Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => (await EmbedWithMetadataAsync(request, headers, ct)).Result;

    public async Task<(OpenAIEmbeddingResponse Result, JsonElement Body, Dictionary<string, string?> Headers)> EmbedWithMetadataAsync(
        OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var key = Key();
        using var client = clients.CreateClient(Id);
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, "v1/embeddings", NativeOptionsPayload.Build(request, Id, Json),
            h => h.Authorization = new("Bearer", key), headers, Json, ct);
        if (Property(body, "data").ValueKind != JsonValueKind.Array) throw new JsonException("DeAPI returned no embeddings.");
        var result = body.Deserialize<OpenAIEmbeddingResponse>(Json) ?? throw new JsonException("DeAPI returned no embeddings.");
        result.Model = String(body, "model") ?? request.Model;
        return (result, body, NativeHttp.SafeResponseMetadataHeaders(upstream));
    }

    private async Task<(string Id, JsonElement Data)> SubmitAsync(string endpoint, HttpContent content, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        var body = await JsonAsync<JsonElement>(client, HttpMethod.Post, V2 + endpoint, content, "Bearer", Key(), headers, ct);
        var data = Property(body, "data");
        var id = String(data, "request_id") ?? String(body, "request_id");
        if (string.IsNullOrWhiteSpace(id)) throw new JsonException("DeAPI did not return a request_id.");
        return (id, data.ValueKind == JsonValueKind.Object ? data : body);
    }

    private async Task<JsonElement> GetJobAsync(string id, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        // Exclude dot segments and delimiters before building an upstream path.
        if (id.Length is 0 or > 256 || id is "." or ".." || id.Any(c => !char.IsAsciiLetterOrDigit(c) && c is not ('-' or '_' or '.')))
            throw new JsonException("DeAPI returned an invalid job ID.");
        using var client = clients.CreateClient(Id);
        var body = await JsonAsync<JsonElement>(client, HttpMethod.Get, $"{V2}jobs/{id}", null, "Bearer", Key(), headers, ct);
        return Property(body, "data") is { ValueKind: JsonValueKind.Object } data ? data : throw new JsonException("DeAPI returned no job status.");
    }

    private async Task<JsonElement> RunJobAsync(string endpoint, HttpContent content, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromMinutes(10));
        try
        {
            var (id, _) = await SubmitAsync(endpoint, content, headers, deadline.Token);
            while (true)
            {
                var job = await GetJobAsync(id, headers, deadline.Token);
                if (State(job) == "done") return job;
                if (State(job) is "error" or "failed" or "cancelled" or "canceled") throw new JsonException(JobError(job));
                await Task.Delay(TimeSpan.FromSeconds(2), deadline.Token);
            }
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested && deadline.IsCancellationRequested)
        {
            throw new TimeoutException("DeAPI job did not finish before the deadline.");
        }
    }

    private static string? State(JsonElement job) => String(job, "status")?.ToLowerInvariant();
    private static string JobError(JsonElement job) => String(job, "error_message") ?? String(job, "error_reason")
        ?? (Property(job, "error_code") is { ValueKind: not JsonValueKind.Undefined and not JsonValueKind.Null } code ? code.ToString() : "DeAPI job failed.");
    private static string? ResultUrl(JsonElement job) => String(job, "result_url") ?? String(Property(job, "result"), "result_url");

    private async Task<(byte[] Bytes, string Mime)> DownloadAsync(string? url, string kind, string fallback, int maxBytes, CancellationToken ct)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) || uri.Scheme != Uri.UriSchemeHttps || !uri.IsDefaultPort
            || uri.HostNameType != UriHostNameType.Dns || uri.UserInfo.Length != 0 || uri.Fragment.Length != 0 || !AllowedAssetHost(uri.IdnHost))
            throw new JsonException("DeAPI returned an invalid or untrusted result URL.");
        // The signed asset URL authenticates the download; do not attach credentials or caller headers.
        using var client = clients.CreateClient(Assets);
        using var request = new HttpRequestMessage(HttpMethod.Get, uri);
        using var response = await NativeHttp.SendAsync(client, request, ct);
        var mime = response.Content.Headers.ContentType?.MediaType;
        if (mime is null or "application/octet-stream") mime = fallback;
        if (!mime.StartsWith(kind + "/", StringComparison.OrdinalIgnoreCase)) throw new JsonException($"DeAPI returned a non-{kind} result.");
        await response.Content.LoadIntoBufferAsync(maxBytes, ct);
        var bytes = await response.Content.ReadAsByteArrayAsync(ct);
        if (bytes.Length == 0) throw new JsonException("DeAPI returned an empty result.");
        return (bytes, mime);
    }

    private bool AllowedAssetHost(string host) => host.Equals("deapi.ai", StringComparison.OrdinalIgnoreCase)
        || host.EndsWith(".deapi.ai", StringComparison.OrdinalIgnoreCase)
        || configuration.GetSection("Providers:deapi:AssetHosts").GetChildren().Any(x => string.Equals(x.Value, host, StringComparison.OrdinalIgnoreCase));

    private static Dictionary<string, JsonElement> Metadata(JsonElement data)
    {
        var value = (JsonObject)Copy(data)!;
        // Do not emit the temporary signed download URL as metadata.
        value.Remove("result_url");
        if (value["result"] is JsonObject result) result.Remove("result_url");
        return NativeMediaOptions.Metadata(Id, value);
    }

    private static JsonContent JsonBody(Dictionary<string, object?> payload) => JsonContent.Create(payload, options: Json);

    private static Dictionary<string, object?> Options(IReadOnlyDictionary<string, JsonElement>? options, params string[] reserved)
        => NativeMediaOptions.Read(Id, options, new HashSet<string>(reserved, StringComparer.OrdinalIgnoreCase))
            .ToDictionary(x => x.Key, x => (object?)x.Value, StringComparer.OrdinalIgnoreCase);

    private static void FormFields(MultipartFormDataContent form, Dictionary<string, object?> payload)
    {
        foreach (var (name, value) in payload) Add(name, JsonSerializer.SerializeToElement(value, Json));
        void Add(string name, JsonElement value)
        {
            if (value.ValueKind == JsonValueKind.Array)
                foreach (var item in value.EnumerateArray()) Add(name, item);
            else if (value.ValueKind != JsonValueKind.Null)
                form.Add(new StringContent(value.ValueKind == JsonValueKind.String ? value.GetString()! : value.GetRawText()), name);
        }
    }
}
