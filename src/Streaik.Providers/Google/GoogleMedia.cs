using System.Net.Http.Json;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class GoogleGeminiProvider : IEmbeddingsProvider, IImageProvider, ISpeechProvider, ITranscriptionProvider, IVideoProvider
{
    private const int InlineLimit = 32 * 1024 * 1024;

    public async Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        request.Model = NormalizeModel(request.Model);
        using var client = CreateClient();
        var key = ApiKey();
        return await NativeHttp.PostAsync<OpenAIEmbeddingRequest, OpenAIEmbeddingResponse>(client, "v1beta/openai/embeddings", request,
            outgoing => outgoing.Authorization = new("Bearer", key), headers, Json, ct);
    }

    public async Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var model = NormalizeModel(request.Model);
        var mime = ImageMime(request.OutputFormat);
        var count = request.N ?? 1;
        var format = GoogleImageFormat(request.Size, mime, request.AdditionalProperties);
        if (model.StartsWith("imagen-", StringComparison.OrdinalIgnoreCase))
        {
            if (count is < 1 or > 4) throw new ProviderRequestException("Imagen requires n between 1 and 4.");
            var predictions = await PostGoogleJsonAsync($"v1beta/models/{model}:predict", new JsonObject
            {
                ["instances"] = new JsonArray(new JsonObject { ["prompt"] = request.Prompt }),
                ["parameters"] = new JsonObject { ["sampleCount"] = count, ["aspectRatio"] = format["aspect_ratio"]?.DeepClone(),
                    ["outputOptions"] = new JsonObject { ["mimeType"] = mime } }
            }, ct);
            return ImageResult(Array(Property(predictions, "predictions")).Select(x => String(x, "bytesBase64Encoded")), mime);
        }
        if (count != 1) throw new ProviderRequestException("Gemini image generation requires n=1.");
        return await GeminiImageAsync(model, new JsonArray(TextPart(request.Prompt)), format, ct);
    }

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var model = NormalizeModel(request.Model);
        if (model.StartsWith("imagen-", StringComparison.OrdinalIgnoreCase) || request.Mask is not null || request.MaskFile is not null || request.N is not (null or 1))
            throw new ProviderRequestException("Use a Gemini image model for unmasked editing with n=1.", 422);
        var input = new JsonArray(TextPart(request.Prompt));
        foreach (var reference in request.Images ?? []) input.Add(GoogleImagePart(reference.ImageUrl, null));
        foreach (var file in request.ImageFiles ?? [])
        {
            await using var source = file.OpenReadStream();
            input.Add(GoogleImagePart(await BoundedBase64Async(source, InlineLimit, ct), file.ContentType));
        }
        if (input.Count == 1) throw new ProviderRequestException("At least one image is required.");
        return await GeminiImageAsync(model, input, GoogleImageFormat(request.Size, ImageMime(request.OutputFormat), request.AdditionalProperties), ct);
    }

    private async Task<OpenAIImagesResponse> GeminiImageAsync(string model, JsonArray input, JsonObject format, CancellationToken ct)
    {
        var body = new JsonObject { ["model"] = model, ["input"] = input, ["response_format"] = format, ["store"] = false };
        var key = ApiKey();
        using var client = CreateClient();
        using var request = new HttpRequestMessage(HttpMethod.Post, InteractionsPath) { Content = JsonContent.Create(body, options: Json) };
        GoogleAuth(key, revision: true)(request.Headers);
        request.Headers.Accept.ParseAdd("application/json");
        using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
        if (!response.IsSuccessStatusCode && _logger.IsEnabled(LogLevel.Debug))
            await LogGeminiImageRejectionAsync(response, model, key, ct);
        response.EnsureSuccessStatusCode();
        var result = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        return ImageResult(OutputParts(result).Where(x => String(x, "type") == "image").Select(x => String(x, "data")),
            OutputParts(result).Where(x => String(x, "type") == "image").Select(x => String(x, "mime_type")).FirstOrDefault() ?? format["mime_type"]!.ToString());
    }

    private async Task LogGeminiImageRejectionAsync(HttpResponseMessage response, string model, string key, CancellationToken ct)
    {
        const int maxErrorBytes = 8192;
        var detail = "No readable Google error message.";
        try
        {
            // Error bodies are untrusted. Read a bounded Google error envelope, never the raw body or the request.
            var buffer = new byte[maxErrorBytes + 1];
            await using var stream = await response.Content.ReadAsStreamAsync(ct);
            var length = await stream.ReadAtLeastAsync(buffer, buffer.Length, false, ct);
            if (length is > 0 and <= maxErrorBytes)
            {
                using var document = JsonDocument.Parse(buffer.AsMemory(0, length));
                if (String(Property(document.RootElement, "error"), "message") is { Length: > 0 } message)
                {
                    message = message.Replace(key, "[redacted]", StringComparison.Ordinal);
                    detail = new string(message.Take(1024).Select(c => char.IsControl(c) ? ' ' : c).ToArray());
                }
            }
        }
        catch (Exception error) when (error is JsonException or IOException or HttpRequestException) { /* Preserve the original HTTP status. */ }
        _logger.LogDebug("Gemini image request for {Model} rejected (HTTP {Status}): {Detail}", model, (int)response.StatusCode, detail);
    }

    private static OpenAIImagesResponse ImageResult(IEnumerable<string?> values, string mime)
    {
        var images = values.Where(x => !string.IsNullOrWhiteSpace(x)).Select(x => new OpenAIImageData { B64Json = x }).ToList();
        if (images.Count == 0) throw new JsonException("Google did not return an image.");
        return new() { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Data = images,
            OutputFormat = mime == "image/jpeg" ? "jpeg" : mime.Split('/')[^1] };
    }

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Speed is not (null or 1)) throw new ProviderRequestException("Gemini speech does not support an explicit speed multiplier.", 422);
        var modelAndVoice = request.Model.Split(':', 2);
        var model = NormalizeModel(modelAndVoice[0]);
        var voice = modelAndVoice.Length == 2 ? modelAndVoice[1] : request.Voice;
        var output = request.ResponseFormat?.ToLowerInvariant();
        if (output is not (null or "pcm" or "wav")) throw new ProviderRequestException("Gemini speech supports pcm or wav on compatible routes.", 422);
        var body = new JsonObject();
        if (request.AdditionalProperties is not null)
            foreach (var (name, value) in request.AdditionalProperties)
                if (name is "generation_config" or "service_tier") body[name] = Copy(value);
        var generation = body["generation_config"] as JsonObject;
        if (generation is null) body["generation_config"] = generation = new JsonObject();
        generation["speech_config"] ??= new JsonArray(new JsonObject { ["voice"] = string.IsNullOrWhiteSpace(voice) ? "Kore" : voice });
        body["model"] = model;
        body["input"] = string.IsNullOrWhiteSpace(request.Instructions) ? request.Input : request.Instructions + "\n\n" + request.Input;
        body["store"] = false;
        body["response_format"] = new JsonObject { ["type"] = "audio" };
        var result = await RunInteractionAsync(body, ct);
        var audioParts = OutputParts(result).Where(x => String(x, "type") == "audio").ToArray();
        var bytes = audioParts.SelectMany(x => Convert.FromBase64String(String(x, "data") ?? throw new JsonException("Missing Google audio data."))).ToArray();
        if (bytes.Length is 0 or > InlineLimit) throw new JsonException("Invalid Google audio output length.");
        var mime = String(audioParts.FirstOrDefault(), "mime_type") ?? "audio/L16";
        if (output == "wav" && !mime.StartsWith("audio/wav", StringComparison.OrdinalIgnoreCase))
        {
            var rate = audioParts.Select(x => String(x, "mime_type")).FirstOrDefault()?.Split(';').Skip(1)
                .Select(x => x.Trim()).FirstOrDefault(x => x.StartsWith("rate=", StringComparison.OrdinalIgnoreCase));
            bytes = WrapPcmAsWave(bytes, rate is not null && int.TryParse(rate[5..], out var samples) && samples is > 0 and <= 192000 ? samples : 24000);
            mime = "audio/wav";
        }
        var message = new HttpResponseMessage(System.Net.HttpStatusCode.OK) { Content = new ByteArrayContent(bytes) };
        message.Content.Headers.ContentType = MediaTypeHeaderValue.TryParse(mime, out var type) ? type : new("audio/L16");
        return message;
    }

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var native = NormalizeModel(model);
        if (IsLiveTranscriptionModel(native)) throw new ProviderRequestException("This Google transcription model requires a Real-time session.", 422);
        if (!mediaType.StartsWith("audio/", StringComparison.OrdinalIgnoreCase)) throw new ProviderRequestException("Gemini transcription requires audio.", 422);
        var prompt = "Generate a transcript of the speech. Do not include any other text.";
        if (!string.IsNullOrWhiteSpace(language)) prompt += " The expected spoken language is " + language + ".";
        if (options?.TryGetValue("prompt", out var hint) == true && hint.ValueKind == JsonValueKind.String) prompt += " Context: " + hint.GetString();
        var body = new JsonObject();
        if (options is not null)
            foreach (var (name, value) in options)
                if (name is "generation_config" or "service_tier") body[name] = Copy(value);
        body["model"] = native;
        body["input"] = new JsonArray(TextPart(prompt), new JsonObject { ["type"] = "audio", ["mime_type"] = mediaType,
            ["data"] = await BoundedBase64Async(audio, InlineLimit, ct) });
        body["store"] = false;
        var result = await RunInteractionAsync(body, ct);
        return new() { Text = InteractionText(result), Language = language, Response = new() { Timestamp = DateTime.UtcNow },
            ProviderMetadata = Property(result, "usage").ValueKind == JsonValueKind.Object ? new() { ["google"] = Property(result, "usage") } : null };
    }

    private Task<JsonElement> RunInteractionAsync(JsonObject body, CancellationToken ct) => PostGoogleJsonAsync(InteractionsPath, body, ct, revision: true);

    private async Task<JsonElement> PostGoogleJsonAsync(string path, JsonObject body, CancellationToken ct, bool revision = false)
    {
        using var client = CreateClient();
        return await NativeHttp.PostAsync<JsonObject, JsonElement>(client, path, body, GoogleAuth(ApiKey(), revision), new Dictionary<string, string>(), Json, ct);
    }

    private static string NormalizeModel(string model) => ResourceId(model.StartsWith("models/", StringComparison.Ordinal) ? model[7..] : model);
    private static JsonObject TextPart(string text) => new() { ["type"] = "text", ["text"] = text };
    private static IEnumerable<JsonElement> OutputParts(JsonElement result) => Array(Property(result, "steps"))
        .Where(step => String(step, "type") == "model_output").SelectMany(step => Array(Property(step, "content"))).Concat(Array(Property(result, "outputs")));
    private static string InteractionText(JsonElement result)
    {
        var parts = OutputParts(result).Where(x => String(x, "type") == "text").Select(x => String(x, "text")).Where(x => x is not null).ToArray();
        return parts.Length > 0 ? string.Concat(parts) : throw new JsonException("Missing Google text output.");
    }

    private static string ImageMime(string? value) => value?.ToLowerInvariant() switch
    {
        null or "png" or "image/png" => "image/png", "jpg" or "jpeg" or "image/jpeg" => "image/jpeg",
        "webp" or "image/webp" => "image/webp", _ => throw new ProviderRequestException("Unsupported image output format.", 422)
    };

    private static JsonObject GoogleImageFormat(string? size, string mime, Dictionary<string, JsonElement>? options)
    {
        var format = new JsonObject { ["type"] = "image", ["mime_type"] = mime };
        if (options?.TryGetValue("aspect_ratio", out var ratio) == true) format["aspect_ratio"] = Copy(ratio);
        if (options?.TryGetValue("image_size", out var imageSize) == true) format["image_size"] = Copy(imageSize);
        if (size is not (null or "auto"))
        {
            if (size.ToUpperInvariant() is "1K" or "2K" or "4K") format["image_size"] ??= size.ToUpperInvariant();
            else if (size.Split('x') is [var w, var h] && int.TryParse(w, out var width) && int.TryParse(h, out var height) && width > 0 && height > 0)
            {
                format["aspect_ratio"] ??= width == height ? "1:1" : width > height ? "3:2" : "2:3";
                format["image_size"] ??= Math.Max(width, height) > 2048 ? "4K" : Math.Max(width, height) >= 2048 ? "2K" : "1K";
            }
            else throw new ProviderRequestException("Use a Google image size (1K, 2K, or 4K) or a width x height pair.", 422);
        }
        return format;
    }

    private static JsonObject GoogleImagePart(string? value, string? mime)
    {
        if (string.IsNullOrWhiteSpace(value)) throw new ProviderRequestException("Missing image reference.");
        if (Uri.TryCreate(value, UriKind.Absolute, out var uri) && uri.Scheme == "https" && uri.UserInfo.Length == 0)
            return new() { ["type"] = "image", ["uri"] = value, ["mime_type"] = mime };
        if (value.StartsWith("data:", StringComparison.Ordinal))
        {
            var comma = value.IndexOf(',');
            if (comma < 0 || !value[..comma].EndsWith(";base64", StringComparison.OrdinalIgnoreCase))
                throw new ProviderRequestException("Expected an image data URL.");
            mime = value[5..(comma - 7)]; value = value[(comma + 1)..];
        }
        if (mime is null || !mime.StartsWith("image/", StringComparison.OrdinalIgnoreCase)) throw new ProviderRequestException("Expected an image MIME type.");
        return new() { ["type"] = "image", ["mime_type"] = mime, ["data"] = value };
    }

    private static byte[] WrapPcmAsWave(byte[] pcm, int rate)
    {
        using var stream = new MemoryStream(pcm.Length + 44);
        using var writer = new BinaryWriter(stream);
        writer.Write("RIFF"u8); writer.Write(pcm.Length + 36); writer.Write("WAVEfmt "u8); writer.Write(16);
        writer.Write((short)1); writer.Write((short)1); writer.Write(rate); writer.Write(rate * 2); writer.Write((short)2); writer.Write((short)16);
        writer.Write("data"u8); writer.Write(pcm.Length); writer.Write(pcm);
        return stream.ToArray();
    }
}
