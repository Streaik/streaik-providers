using System.Globalization;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class AetherProviderRegistration
{
    public static IServiceCollection AddAether(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AetherProvider.Id, "https://api.aetherapi.dev/v1/");
        PublicMediaDelivery.AddClient(services, AetherProvider.Delivery);
        services.AddMemoryCache();
        return services.AddProvider<AetherProvider>(AetherProvider.Id);
    }
}

internal sealed class AetherProvider(IHttpClientFactory clients, IProviderCredentials credentials, IMemoryCache cache)
    : CompatibleChatEmbeddingProvider(clients, credentials, Id, ""), IResponsesProvider, IImageModelRoutingProvider,
      IStreamingImagesProvider, IStreamingSpeechProvider, ITranscriptionProvider
{
    internal const string Id = "aether";
    internal const string Delivery = "aether-delivery";
    private const int ImageLimit = 24 * 1024 * 1024;
    private readonly BearerTextEndpoints text = new(clients, credentials, Id);
    private static readonly HashSet<string> ImageReserved = new(StringComparer.OrdinalIgnoreCase)
        { "model", "prompt", "n", "size", "style", "quality", "response_format", "stream", "partial_images", "seed", "aspect_ratio", "image", "images", "mask" };
    private static readonly HashSet<string> SpeechReserved = new(StringComparer.OrdinalIgnoreCase)
        { "model", "input", "voice", "response_format", "stream_format", "stream", "speed" };
    private static readonly HashSet<string> TranscriptionReserved = new(StringComparer.OrdinalIgnoreCase) { "model", "file", "stream" };

    public Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.CreateResponseAsync(request, headers, ct);
    public IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.StreamResponseAsync(request, headers, ct);

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var key = ConfiguredApiKey();
        if (string.IsNullOrWhiteSpace(key)) yield break;
        var body = await FetchModelsAsync(key, ct);
        foreach (var item in Array(Property(body, "data")))
        {
            if (String(item, "id") is not { Length: > 0 } id) continue;
            var context = Property(item, "context");
            var input = ProviderPrices.TokenPrice(Property(item, "input_cost"));
            var output = ProviderPrices.TokenPrice(Property(item, "output_cost"));
            yield return new Model { Id = id, Name = String(item, "name") ?? id, OwnedBy = String(item, "owned_by") ?? Id,
                Type = IsImageId(id) ? "image" : String(item, "type") ?? "model", Description = String(item, "description"),
                ContextWindow = int.TryParse(context.ToString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var size) ? size : null,
                Pricing = input > 0 && output > 0 ? new ModelPricing { Input = input.Value / 1_000_000m, Output = output.Value / 1_000_000m } : null };
        }
    }

    protected override async Task<JsonElement> FetchModelsAsync(string key, CancellationToken ct)
    {
        var cacheKey = (Id, "models", Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(key))));
        if (cache.TryGetValue(cacheKey, out JsonElement cached)) return cached;
        var models = await base.FetchModelsAsync(key, ct);
        return cache.Set(cacheKey, models, TimeSpan.FromMinutes(15));
    }

    public async Task<bool> IsImageModelAsync(string model, CancellationToken ct)
    {
        if (IsImageId(model)) return true;
        try
        {
            await foreach (var entry in ListModelsAsync(ct))
                if (entry.Id.Equals(model, StringComparison.OrdinalIgnoreCase)) return entry.Type is "image" or "image-generation";
        }
        catch (Exception ex) when (!ct.IsCancellationRequested && ex is HttpRequestException or JsonException)
        {
            // Text turns still work with keys that cannot read the catalog.
        }
        return false;
    }

    private static bool IsImageId(string id) => id.Split('/').Last() is var name && (name.StartsWith("dall-e", StringComparison.OrdinalIgnoreCase)
        || name.StartsWith("gpt-image", StringComparison.OrdinalIgnoreCase) || name.StartsWith("imagen", StringComparison.OrdinalIgnoreCase)
        || name.StartsWith("flux", StringComparison.OrdinalIgnoreCase) || name.StartsWith("stable-diffusion", StringComparison.OrdinalIgnoreCase));

    public async Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Prompt) || request.N is <= 0 or > 10) throw new ProviderRequestException("Invalid Aether image prompt or count.");
        var payload = NativeMediaOptions.Read(Id, request.AdditionalProperties, ImageReserved).ToDictionary(p => p.Key, p => (object?)p.Value);
        payload["model"] = request.Model;
        payload["prompt"] = request.Prompt;
        payload["response_format"] = "b64_json";
        if (request.N is { } n) payload["n"] = n;
        if (request.Size is { } size) payload["size"] = size;
        if (request.Style is { } style) payload["style"] = style;
        if (request.Quality is { } quality) payload["quality"] = quality;
        using var client = CreateClient();
        var body = await NativeHttp.PostAsync<object, JsonElement>(client, "images/generations", payload, Bearer(ApiKey()), headers, Json, ct);
        var data = Property(body, "data");
        if (data.ValueKind != JsonValueKind.Array || data.GetArrayLength() is < 1 or > 10) throw new JsonException("Missing or invalid Aether image data.");
        var result = body.Deserialize<OpenAIImagesResponse>(Json) ?? throw new JsonException("Missing Aether image response.");
        var images = new List<OpenAIImageData>();
        var remaining = ImageLimit;
        using var delivery = CreateClient(Delivery);
        foreach (var item in data.EnumerateArray())
        {
            var encoded = String(item, "b64_json");
            if (string.IsNullOrWhiteSpace(encoded) && String(item, "url") is { Length: > 0 } url)
            {
                var downloaded = await PublicMediaDelivery.DownloadAsync(delivery, url, "image/png", remaining, ct);
                encoded = downloaded.Data;
                result.OutputFormat ??= downloaded.Mime.Split('/')[1];
            }
            if (string.IsNullOrWhiteSpace(encoded)) throw new JsonException("Aether returned no image content.");
            remaining -= encoded.Length / 4 * 3 - (encoded.EndsWith("==", StringComparison.Ordinal) ? 2 : encoded.EndsWith('=') ? 1 : 0);
            if (remaining < 0) throw new ProviderRequestException("Image output exceeds the inline size limit.", 413);
            images.Add(new() { B64Json = encoded, RevisedPrompt = String(item, "revised_prompt") });
        }
        result.Data = images;
        result.OutputFormat ??= "png";
        // Keep native metadata, without duplicating the inline image bytes.
        var metadata = ToObject(body);
        foreach (var item in metadata["data"]!.AsArray().OfType<System.Text.Json.Nodes.JsonObject>()) item.Remove("b64_json");
        (result.AdditionalProperties ??= new())["providerMetadata"] = JsonSerializer.SerializeToElement(NativeMediaOptions.Metadata(Id, metadata), Json);
        return result;
    }

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => throw new ProviderRequestException("Aether does not provide image editing.", 422);

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageGenerationRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        if (request.N is > 1) throw new ProviderRequestException("Streaming images require one output.");
        yield return NativeMediaOptions.CompletedImage(await GenerateImageAsync(request, headers, ct), edit: false);
    }

    public IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => throw new ProviderRequestException("Aether does not provide image editing.", 422);

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Input) || string.IsNullOrWhiteSpace(request.Voice))
            throw new ProviderRequestException("Aether speech requires input and voice.");
        var payload = NativeMediaOptions.Read(Id, request.AdditionalProperties, SpeechReserved).ToDictionary(p => p.Key, p => (object?)p.Value);
        payload["model"] = request.Model;
        payload["input"] = request.Input;
        payload["voice"] = request.Voice;
        if (request.ResponseFormat is { } format) payload["response_format"] = format;
        if (request.Speed is { } speed) payload["speed"] = speed;
        using var client = CreateClient();
        using var result = await SendAsync(client, HttpMethod.Post, "audio/speech", JsonContent.Create(payload, options: Json), "Bearer", ApiKey(), headers, ct);
        var audio = result.Content;
        audio.Headers.ContentType ??= new(request.ResponseFormat switch
        { "opus" => "audio/opus", "aac" => "audio/aac", "flac" => "audio/flac", "wav" => "audio/wav", "pcm" => "application/octet-stream", _ => "audio/mpeg" });
        var upstream = result.Headers.Concat(audio.Headers).ToDictionary(h => h.Key, h => (string?)string.Join(", ", h.Value), StringComparer.OrdinalIgnoreCase);
        var metadata = NativeMediaOptions.Metadata(Id, new { headers = NativeHttp.SafeResponseMetadataHeaders(upstream) });
        result.Content = null; // Transfer ownership to the gateway's response.
        return new SpeechProviderResponse(audio, metadata);
    }

    public Task<HttpResponseMessage> SpeakStreamingAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeakAsync(request, headers, ct);

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName, string? language,
        IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var fields = NativeMediaOptions.Read(Id, options, TranscriptionReserved);
        if (fields.Remove("responseFormat", out var format)) fields.TryAdd("response_format", format);
        fields.TryAdd("response_format", JsonSerializer.SerializeToElement("json"));
        if (!string.IsNullOrWhiteSpace(language)) fields["language"] = JsonSerializer.SerializeToElement(language);
        using var form = new MultipartFormDataContent();
        AddFile(form, "file", audio, mediaType, fileName);
        form.Add(new StringContent(model), "model");
        foreach (var (name, value) in fields)
            if (value.ValueKind is not (JsonValueKind.Null or JsonValueKind.Undefined))
                form.Add(new StringContent(value.ValueKind == JsonValueKind.String ? value.GetString()! : value.GetRawText()), name);
        using var client = CreateClient();
        using var upstream = await SendAsync(client, HttpMethod.Post, "audio/transcriptions", form, "Bearer", ApiKey(), headers, ct);
        if (upstream.Content.Headers.ContentType?.MediaType is "text/plain" or "text/vtt")
        {
            var textResult = await upstream.Content.ReadAsStringAsync(ct);
            return new() { Text = textResult, Segments = [], ProviderMetadata = NativeMediaOptions.Metadata(Id, textResult),
                Response = new() { ModelId = model, Body = textResult, Timestamp = DateTime.UtcNow } };
        }
        var body = await NativeHttp.ReadAsync<JsonElement>(upstream, Json, ct);
        var segments = Array(Property(body, "segments")).Select(item =>
        {
            var start = Number(item, "start", "start_second", "startSecond") ?? 0;
            return new TranscriptionSegment { Text = String(item, "text") ?? "", StartSecond = start,
                EndSecond = Math.Max(start, Number(item, "end", "end_second", "endSecond") ?? 0) };
        }).ToArray();
        return new() { Text = String(body, "text") ?? string.Join(' ', segments.Select(s => s.Text)), Segments = segments,
            Language = String(body, "language"), DurationInSeconds = Number(body, "duration") ?? Number(Property(body, "usage"), "seconds"),
            ProviderMetadata = NativeMediaOptions.Metadata(Id, body),
            Response = new() { ModelId = String(body, "model") ?? model, Timestamp = DateTime.UtcNow, Body = body } };
    }

    private static float? Number(JsonElement body, params string[] names)
    {
        foreach (var name in names)
            if (Property(body, name) is { ValueKind: JsonValueKind.Number } value && value.TryGetSingle(out var number) && float.IsFinite(number)) return number;
        return null;
    }
}
