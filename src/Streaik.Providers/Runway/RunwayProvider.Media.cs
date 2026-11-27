using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class RunwayProvider
{
    private const string SoundModel = "eleven_text_to_sound_v2";

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => ImageAsync(request.Model, request.Prompt, request.Size, request.AdditionalProperties, null, headers, ct);

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Mask is not null || request.MaskFile is not null) throw new ProviderRequestException("Runway does not accept image masks.", 422);
        var references = new JsonArray();
        var remaining = InlineLimit;
        foreach (var image in request.Images ?? [])
        {
            if (image.FileId is not null || string.IsNullOrWhiteSpace(image.ImageUrl))
                throw new ProviderRequestException("Runway requires image URLs or inline images.", 422);
            var uri = MediaUri(image.ImageUrl, "image/png");
            if (uri.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
            {
                remaining -= uri.Length / 4 * 3;
                if (remaining < 0) throw new ProviderRequestException("Runway image input is too large.", 413);
            }
            references.Add(new JsonObject { ["uri"] = uri });
        }
        foreach (var file in request.ImageFiles ?? [])
        {
            if (file.ContentType is not ("image/png" or "image/jpeg" or "image/webp" or "image/gif"))
                throw new ProviderRequestException("Runway requires an image upload.", 415);
            await using var stream = file.OpenReadStream();
            var data = await BoundedBase64Async(stream, remaining, ct);
            remaining -= Base64Size(data);
            references.Add(new JsonObject { ["uri"] = $"data:{file.ContentType};base64,{data}" });
        }
        return await ImageAsync(request.Model, request.Prompt, request.Size, request.AdditionalProperties, references, headers, ct);
    }

    private async Task<OpenAIImagesResponse> ImageAsync(string model, string prompt, string? size, Dictionary<string, JsonElement>? options,
        JsonArray? references, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(prompt)) throw new ProviderRequestException("Runway requires an image prompt.");
        var payload = Options(options, "model", "prompt", "promptText", "n", "size", "aspect_ratio", "stream", "partial_images", "image", "images", "mask");
        payload["model"] = model;
        payload["promptText"] = prompt;
        if (payload["ratio"] is null)
            payload["ratio"] = Ratio(size ?? (options?.GetValueOrDefault("aspect_ratio") is { ValueKind: JsonValueKind.String } aspect ? aspect.GetString() : null) ?? "1024:1024");
        if (references is { Count: > 0 }) payload["referenceImages"] = references;
        if (payload["contentModeration"] is { } moderation && (moderation is not JsonObject obj
            || obj["publicFigureThreshold"] is { } threshold && threshold.GetValueKind() != JsonValueKind.String))
            throw new ProviderRequestException("Runway contentModeration must be an object with a string publicFigureThreshold.");
        var (created, _) = await SubmitAsync("text_to_image", payload, headers, ct);
        var task = await WaitAsync(TaskId(created), ct);
        var result = new OpenAIImagesResponse { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Data = [],
            AdditionalProperties = new() { ["providerMetadata"] = JsonSerializer.SerializeToElement(NativeMediaOptions.Metadata(Id, task), Json) } };
        var remaining = InlineLimit;
        foreach (var url in Outputs(task))
        {
            var (data, mime) = await DownloadAsync(url, "image/png", remaining, ct);
            remaining -= Base64Size(data);
            result.OutputFormat ??= mime.Split('/')[1];
            result.Data.Add(new() { B64Json = data });
        }
        if (result.Data.Count == 0) throw new JsonException("Runway returned no image output.");
        return result;
    }

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageGenerationRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        if (request.N is > 1) throw new ProviderRequestException("Streaming images require one output.");
        yield return NativeMediaOptions.CompletedImage(await GenerateImageAsync(request, headers, ct), edit: false);
    }

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageEditRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        if (request.N is > 1) throw new ProviderRequestException("Streaming images require one output.");
        yield return NativeMediaOptions.CompletedImage(await EditImageAsync(request, headers, ct), edit: true);
    }

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Input)) throw new ProviderRequestException("Runway requires speech input text.");
        var payload = Options(request.AdditionalProperties, "model", "input", "promptText", "response_format", "stream_format", "stream", "language");
        var voiceNode = payload["voice"];
        var soundNode = payload["soundEffects"];
        payload.Remove("voice");
        payload.Remove("soundEffects");
        payload["model"] = request.Model;
        payload["promptText"] = request.Input;
        var effect = request.Model.Equals(SoundModel, StringComparison.OrdinalIgnoreCase);
        if (effect)
        {
            if (soundNode is not null && soundNode is not JsonObject) throw new ProviderRequestException("Runway soundEffects must be an object.");
            if (soundNode is JsonObject sound)
                foreach (var (key, value) in sound)
                    if (key is not ("model" or "input" or "promptText" or "voice")) payload[key] = value?.DeepClone();
            if (payload["duration"] is { } duration && (duration is not JsonValue number || !number.TryGetValue<double>(out var seconds)
                || !double.IsFinite(seconds) || seconds is < 0.5 or > 30))
                throw new ProviderRequestException("Runway sound effect duration must be between 0.5 and 30 seconds.");
            if (payload["loop"] is { } loop && loop.GetValueKind() is not (JsonValueKind.True or JsonValueKind.False))
                throw new ProviderRequestException("Runway sound effect loop must be a boolean.");
        }
        else
        {
            if (voiceNode is not null && voiceNode is not JsonObject) throw new ProviderRequestException("Runway voice options must be an object.");
            var voice = voiceNode as JsonObject ?? new JsonObject();
            var preset = request.Voice ?? (voice["presetId"] is JsonValue value && value.TryGetValue<string>(out var nativePreset) ? nativePreset : null);
            if (string.IsNullOrWhiteSpace(preset)) throw new ProviderRequestException("Runway speech requires a voice preset ID.");
            voice["type"] = "runway-preset";
            voice["presetId"] = preset.Trim();
            payload["voice"] = voice;
        }
        var (created, _) = await SubmitAsync(effect ? "sound_effect" : "text_to_speech", payload, headers, ct);
        var task = await WaitAsync(TaskId(created), ct);
        var url = Outputs(task).FirstOrDefault() ?? throw new JsonException("Runway returned no audio output.");
        var (data, mime) = await DownloadAsync(url, "audio/mpeg", InlineLimit, ct);
        var content = new ByteArrayContent(Convert.FromBase64String(data));
        content.Headers.ContentType = new(mime);
        return new SpeechProviderResponse(content, NativeMediaOptions.Metadata(Id, task));
    }

    public Task<HttpResponseMessage> SpeakStreamingAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeakAsync(request, headers, ct);

    private static string Ratio(string size) => size.Replace('x', ':').Replace('X', ':');

    private static int Base64Size(string data) => data.Length / 4 * 3 - (data.EndsWith("==", StringComparison.Ordinal) ? 2 : data.EndsWith('=') ? 1 : 0);

    private static string MediaUri(string source, string mediaType)
    {
        if (source.StartsWith("data:", StringComparison.OrdinalIgnoreCase) || source.StartsWith("runway://", StringComparison.OrdinalIgnoreCase)) return source;
        if (Uri.TryCreate(source, UriKind.Absolute, out var url))
        {
            if (url.Scheme == Uri.UriSchemeHttps && url.UserInfo.Length == 0) return source;
            throw new ProviderRequestException("Runway media references require HTTPS, runway://, or inline data.");
        }
        return $"data:{mediaType};base64,{source}";
    }
}
