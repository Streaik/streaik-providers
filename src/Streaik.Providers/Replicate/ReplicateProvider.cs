using System.Globalization;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.Catalog;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class ReplicateProviderRegistration
{
    public static IServiceCollection AddReplicate(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, ReplicateProvider.Id, "https://api.replicate.com/v1/");
        PublicMediaDelivery.AddClient(services, ReplicateProvider.Assets);
        return services.AddProvider<ReplicateProvider>(ReplicateProvider.Id);
    }
}

internal sealed class ReplicateProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IImageModelRoutingProvider, IStreamingImagesProvider
{
    internal const string Id = "replicate";
    internal const string Assets = "replicate-assets";
    private const int MaxImageBytes = 16 * 1024 * 1024;
    private static readonly Lazy<HashSet<string>> ImageModels = new(() => ProviderCatalog.ReadModels(Id)!
        .Deserialize<ModelResponse>(Json)!.Data.Select(model => model.Id[(Id.Length + 1)..]).ToHashSet(StringComparer.OrdinalIgnoreCase));
    private static readonly HashSet<string> Reserved = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "version", "input", "webhook", "webhook_events_filter", "stream", "partial_images", "response_format",
        "n", "size", "images", "image[]", "files", "mask", "headers", "cancel-after"
    };

    public IAsyncEnumerable<Model> ListModelsAsync(CancellationToken ct) => StaticModelListing.ListAsync(Id, credentials.GetApiKey(Id), ct);
    public Task<bool> IsImageModelAsync(string model, CancellationToken ct) => Task.FromResult(ImageModels.Value.Contains(model));

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PredictAsync(request.Model, request.Prompt, request.Size, request.N, request.OutputFormat, request.AdditionalProperties, null, false, headers, ct);

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => PredictAsync(request.Model, request.Prompt, request.Size, request.N, request.OutputFormat, request.AdditionalProperties, request, false, headers, ct);

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        yield return NativeMediaOptions.CompletedImage(await PredictAsync(request.Model, request.Prompt, request.Size, request.N,
            request.OutputFormat, request.AdditionalProperties, null, true, headers, ct), edit: false);
    }

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        yield return NativeMediaOptions.CompletedImage(await PredictAsync(request.Model, request.Prompt, request.Size, request.N,
            request.OutputFormat, request.AdditionalProperties, request, true, headers, ct), edit: true);
    }

    private async Task<OpenAIImagesResponse> PredictAsync(string model, string prompt, string? size, int? count, string? format,
        Dictionary<string, JsonElement>? extensions, OpenAIImageEditRequest? edit, bool stream, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!ImageModels.Value.TryGetValue(model, out var nativeModel)) throw new ProviderRequestException("Unsupported Replicate image model.", 422);
        if (string.IsNullOrWhiteSpace(prompt)) throw new ProviderRequestException("Replicate requires a prompt.");
        var native = NativeMediaOptions.Read(Id, extensions, Reserved);
        var preferWait = Remove("preferWaitSeconds");
        var wait = 60;
        if (preferWait is { ValueKind: not JsonValueKind.Null } waitOption)
        {
            if (!int.TryParse(waitOption.ToString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out wait))
                throw new ProviderRequestException("Replicate preferWaitSeconds must be an integer.");
            wait = Math.Clamp(wait, 1, 60);
        }
        var cancelAfter = Remove("cancelAfter");
        if (cancelAfter is { ValueKind: not (JsonValueKind.Null or JsonValueKind.String) })
            throw new ProviderRequestException("Replicate cancelAfter must be a duration string.");
        var overrides = Remove("inputOverrides");
        var ratio = Remove("aspect_ratio");
        var (width, height) = Dimensions(size, ratio?.ToString());
        var input = new Dictionary<string, object?> { ["prompt"] = prompt, ["width"] = width, ["height"] = height };
        if (format is not null) input["output_format"] = format;
        var warnings = new List<object>();
        if (count > 1) Warn("n");
        if (edit is not null)
        {
            if ((edit.ImageFiles?.Length ?? 0) + (edit.Images?.Length ?? 0) > 1) Warn("files");
            if (edit.MaskFile is not null || edit.Mask is not null) Warn("mask");
            if (edit.ImageFiles?.FirstOrDefault() is { } upload)
            {
                if (!System.Net.Http.Headers.MediaTypeHeaderValue.TryParse(upload.ContentType, out var mime)
                    || !mime.MediaType!.StartsWith("image/", StringComparison.OrdinalIgnoreCase) || upload.Length == 0)
                    throw new ProviderRequestException("Replicate requires a nonempty image upload.");
                await using var content = upload.OpenReadStream();
                input["file"] = $"data:{mime.MediaType};base64,{await BoundedBase64Async(content, MaxImageBytes, ct)}";
            }
            else if (edit.Images?.FirstOrDefault()?.ImageUrl is { Length: > 0 } url)
                input["file"] = url.StartsWith("data:image/", StringComparison.OrdinalIgnoreCase) ? url : PublicMediaDelivery.RequirePublicHttpsUri(url).AbsoluteUri;
            else throw new ProviderRequestException("Replicate requires an inline image, upload, or HTTPS image URL.");
        }
        foreach (var (name, value) in native) input[name] = value;
        if (overrides is { ValueKind: not JsonValueKind.Null } fields)
            foreach (var (name, value) in NativeMediaOptions.Read(Id, new Dictionary<string, JsonElement> { [Id] = fields }, Reserved))
                if (name is not ("cancelAfter" or "preferWaitSeconds" or "inputOverrides")) input[name] = value;

        var key = RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromMinutes(3));
        var (prediction, upstreamHeaders) = await RunPredictionAsync(client, nativeModel, input, wait, cancelAfter?.ToString(), key, headers, deadline.Token);
        var urls = ImageUrls(Property(prediction, "output")).Distinct(StringComparer.Ordinal).Take(17).ToArray();
        if (urls.Length is 0 or > 16) throw new JsonException("Replicate returned an invalid number of images.");
        if (stream && urls.Length != 1) throw new ProviderRequestException("Image streaming requires a single base64 output.", 502);
        var data = new List<OpenAIImageData>();
        using var assets = clients.CreateClient(Assets);
        var remaining = MaxImageBytes;
        foreach (var url in urls)
        {
            var (image, mime) = await PublicMediaDelivery.DownloadAsync(assets, url, FallbackMime(url, format), remaining, deadline.Token);
            if (image.Length == 0) throw new IOException("Replicate returned an empty image.");
            remaining -= image.Length / 4 * 3 - (image.EndsWith("==", StringComparison.Ordinal) ? 2 : image.EndsWith('=') ? 1 : 0);
            data.Add(new() { B64Json = image });
            if (data.Count == 1) format = mime[6..] == "svg+xml" ? "svg" : mime[6..];
        }
        var details = ToObject(prediction);
        // Do not echo submitted files, asset URLs or the provider's prediction control links.
        foreach (var name in new[] { "input", "output", "urls" }) details.Remove(name);
        return new() { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Data = data, OutputFormat = format, Size = size,
            AdditionalProperties = new() { ["warnings"] = JsonSerializer.SerializeToElement(warnings, Json),
                ["providerMetadata"] = JsonSerializer.SerializeToElement(NativeMediaOptions.Metadata(Id,
                    new { prediction = details, headers = upstreamHeaders }), Json) } };

        JsonElement? Remove(string name)
        {
            var match = native.Keys.FirstOrDefault(field => field.Equals(name, StringComparison.OrdinalIgnoreCase));
            return match is not null && native.Remove(match, out var element) ? element : null;
        }
        void Warn(string feature) => warnings.Add(new { type = "unsupported", feature });
    }

    private static async Task<(JsonElement Prediction, Dictionary<string, string?> Headers)> RunPredictionAsync(HttpClient client, string model,
        Dictionary<string, object?> input, int wait, string? cancelAfter, string key, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var start = new HttpRequestMessage(HttpMethod.Post, $"models/{model}/predictions")
        { Content = JsonContent.Create(new { input }, options: Json) };
        NativeHttp.ApplyHeaders(start, headers);
        start.Headers.Authorization = new("Bearer", key);
        start.Headers.TryAddWithoutValidation("Prefer", $"wait={wait.ToString(CultureInfo.InvariantCulture)}");
        if (!string.IsNullOrWhiteSpace(cancelAfter))
        {
            if (cancelAfter.Length > 64 || cancelAfter.Any(c => !char.IsAsciiLetterOrDigit(c) && c != '.'))
                throw new ProviderRequestException("Invalid Replicate cancelAfter duration.");
            start.Headers.Add("Cancel-After", cancelAfter);
        }
        using var response = await NativeHttp.SendAsync(client, start, ct);
        var prediction = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        var safe = NativeHttp.SafeResponseMetadataHeaders(response.Headers.Concat(response.Content.Headers)
            .GroupBy(header => header.Key, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => (string?)string.Join(", ", group.SelectMany(header => header.Value)), StringComparer.OrdinalIgnoreCase));
        var id = String(prediction, "id");
        var delayMs = 1000;
        while (true)
        {
            var status = String(prediction, "status");
            if (status is "failed" or "canceled" or "aborted")
                throw new ProviderRequestException($"Replicate prediction {status}.", 502);
            if (status == "succeeded" || status is null && Property(prediction, "output").ValueKind is not (JsonValueKind.Null or JsonValueKind.Undefined))
                return (prediction, safe);
            if (id is not { Length: > 0 and <= 128 } || id.Any(c => !char.IsAsciiLetterOrDigit(c) && c is not ('_' or '-')))
                throw new JsonException("Replicate returned an invalid prediction ID.");
            await Task.Delay(delayMs, ct);
            prediction = await NativeHttp.GetAsync<JsonElement>(client, $"predictions/{id}", NativeHttp.Bearer(key), ct);
            delayMs = Math.Min(5000, delayMs * 3 / 2);
        }
    }

    private static IEnumerable<string> ImageUrls(JsonElement output) => output.ValueKind switch
    {
        JsonValueKind.Array => Array(output).SelectMany(ImageUrls),
        JsonValueKind.Object => output.EnumerateObject().SelectMany(property => ImageUrls(property.Value)),
        JsonValueKind.String when output.GetString() is { } url && Uri.IsWellFormedUriString(url, UriKind.Absolute) => [url],
        _ => []
    };

    private static string FallbackMime(string url, string? format)
        => (Path.GetExtension(PublicMediaDelivery.RequirePublicHttpsUri(url).AbsolutePath).TrimStart('.').ToLowerInvariant(), format?.ToLowerInvariant()) switch
        {
            ("jpg" or "jpeg", _) or ("", "jpg" or "jpeg") => "image/jpeg",
            ("webp", _) or ("", "webp") => "image/webp",
            ("svg", _) or ("", "svg") => "image/svg+xml",
            _ => "image/png"
        };

    private static (int Width, int Height) Dimensions(string? size, string? aspectRatio)
    {
        if (size?.ToLowerInvariant().Split('x', ':') is [var w, var h] && int.TryParse(w, out var width) && int.TryParse(h, out var height) && width > 0 && height > 0)
            return (width, height);
        if (aspectRatio?.Split(':') is [var a, var b] && int.TryParse(a, out var x) && int.TryParse(b, out var y) && x > 0 && y > 0)
        {
            var ratio = (double)x / y;
            if (Math.Round(1920 / ratio) is >= 1024 and <= 1080 and var inferredHeight) return (1920, (int)inferredHeight);
            if (Math.Round(1080 * ratio) is >= 1024 and <= 1920 and var inferredWidth) return ((int)inferredWidth, 1080);
        }
        return (1024, 1024);
    }
}
