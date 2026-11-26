using System.Globalization;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class StabilityAIRegistration
{
    public static IServiceCollection AddStabilityAI(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, "stabilityai", "https://api.stability.ai/v2beta/");
        return services.AddProvider<StabilityAIProvider>("stabilityai");
    }
}

internal sealed class StabilityAIProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IImageModelRoutingProvider, ISpeechProvider, IChatAudioRoutingProvider, IStreamingSpeechEventsProvider
{
    private const string Id = "stabilityai";
    private const int MediaLimit = 25 * 1024 * 1024;
    private static readonly string[] Ratios = ["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"];

    public IAsyncEnumerable<Model> ListModelsAsync(CancellationToken ct) => StaticModelListing.ListAsync(Id, credentials.GetApiKey(Id), ct);
    public Task<bool> IsImageModelAsync(string model, CancellationToken ct) => Task.FromResult(ImageEndpoint(model) is not null);
    public bool IsSpeechModel(string model) => model is "stable-audio-2" or "stable-audio-2.5" or "stable-audio-3";
    public bool IsTranscriptionModel(string model) => false;

    public async Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var endpoint = ImageEndpoint(request.Model) ?? throw new ProviderRequestException("Unknown Stability AI image model.", 422);
        var options = Options(request.AdditionalProperties, "mode", "image", "images", "files", "mask", "strength", "size", "n", "partial_images");
        var format = (request.OutputFormat ?? Text(options, "output_format") ?? "png").Trim().ToLowerInvariant();
        format = format switch { "jpg" => "jpeg", "png" or "jpeg" or "webp" => format, _ => throw new ProviderRequestException("Stability AI supports PNG, JPEG, or WebP.", 422) };
        var ratio = string.IsNullOrWhiteSpace(request.Size) || request.Size == "auto" ? Text(options, "aspect_ratio") ?? "1:1" : AspectRatio(request.Size);
        if (!Ratios.Contains(ratio, StringComparer.Ordinal)) throw new ProviderRequestException("Unsupported Stability AI aspect ratio.", 422);
        options["prompt"] = request.Prompt;
        options["aspect_ratio"] = ratio;
        options["output_format"] = format;
        if (endpoint.EndsWith("/sd3", StringComparison.Ordinal))
        {
            options["mode"] = "text-to-image";
            options["model"] = request.Model.ToLowerInvariant();
        }

        using var response = await SubmitAsync(endpoint, options, "image/*", headers, ct);
        format = response.Content.Headers.ContentType?.MediaType switch
        {
            "image/png" => "png", "image/jpeg" => "jpeg", "image/webp" => "webp", null or "application/octet-stream" => format,
            _ => throw new IOException("Stability AI did not return an image.")
        };
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        var image = await BoundedBase64Async(stream, MediaLimit, ct);
        if (image.Length == 0) throw new IOException("Stability AI returned an empty image.");
        var extras = new Dictionary<string, JsonElement> { ["providerMetadata"] = JsonSerializer.SerializeToElement(Metadata(response), Json) };
        if (request.N is > 1) extras["warnings"] = JsonSerializer.SerializeToElement(new[] { new { type = "unsupported", feature = "n" } }, Json);
        return new() { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Size = request.Size, OutputFormat = format,
            Data = [new() { B64Json = image }], AdditionalProperties = extras };
    }

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => throw new ProviderRequestException("Stability AI does not support image editing via this provider.", 422);

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!IsSpeechModel(request.Model)) throw new ProviderRequestException("Unknown Stable Audio model.", 422);
        if (string.IsNullOrWhiteSpace(request.Input)) throw new ProviderRequestException("A Stable Audio prompt is required.");
        var options = Options(request.AdditionalProperties, "voice", "language", "instructions", "speed", "stream_format");
        var format = (request.ResponseFormat ?? Text(options, "output_format") ?? "mp3").Trim().ToLowerInvariant();
        var mime = format switch { "mp3" => "audio/mpeg", "wav" => "audio/wav",
            _ => throw new ProviderRequestException("Stable Audio supports MP3 or WAV.", 422) };
        options["model"] = request.Model;
        options["prompt"] = request.Input;
        options["output_format"] = format;
        // AI Happy sends the selected model in the form to this shared text-to-audio endpoint.
        using var response = await SubmitAsync("audio/stable-audio-2/text-to-audio", options, "audio/*", headers, ct);
        var actual = response.Content.Headers.ContentType?.MediaType;
        if (actual is not (null or "application/octet-stream") && !actual.StartsWith("audio/", StringComparison.OrdinalIgnoreCase))
            throw new IOException("Stability AI did not return audio.");
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        using var audio = new MemoryStream();
        var buffer = new byte[16 * 1024];
        while (await stream.ReadAsync(buffer, ct) is > 0 and var read)
        {
            if (audio.Length + read > MediaLimit) throw new IOException("Stable Audio output is too large.");
            audio.Write(buffer, 0, read);
        }
        if (audio.Length == 0) throw new IOException("Stability AI returned empty audio.");
        var content = new ByteArrayContent(audio.GetBuffer(), 0, (int)audio.Length);
        content.Headers.ContentType = new(actual is null or "application/octet-stream" ? mime : actual);
        return new SpeechProviderResponse(content, Metadata(response));
    }

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Text is not { Length: > 0 } || request.Text.Any(part => part is null || part.Type != "text" || part.Text is null || part.ProviderOptions is not null)
            || request.OutputFormat?.Rate is not null)
            throw new ProviderRequestException("Stable Audio does not support per-part options or a custom sample rate.", 422);
        return StreamSpeechAsync(new AudioSpeechRequest { Model = request.Model, Input = string.Concat(request.Text.Select(part => part.Text)),
            Voice = request.Voice ?? request.Text.FirstOrDefault(part => part.Voice is not null)?.Voice,
            ResponseFormat = request.OutputFormat?.Type, Instructions = request.Instructions, Speed = request.Speed,
            AdditionalProperties = request.ProviderOptions?.TryGetValue(Id, out var native) == true ? new() { [Id] = native } : null }, headers, ct);
    }

    public async IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        // The upstream returns one completed audio file, not incremental synthesis events.
        using var response = await SpeakAsync(request, headers, ct);
        var audio = await response.Content.ReadAsByteArrayAsync(ct);
        var mime = response.Content.Headers.ContentType!.MediaType!;
        var warnings = new List<object>();
        foreach (var (field, unsupported) in new[] { ("voice", request.Voice is not null), ("instructions", request.Instructions is not null), ("speed", request.Speed is not null) })
            if (unsupported) warnings.Add(new { type = "unsupported", feature = field });
        yield return new SpeechStreamStartPart { Warnings = warnings };
        yield return new SpeechResponseMetadataPart { ModelId = request.Model, Timestamp = DateTime.UtcNow };
        yield return new SpeechAudioPart { Audio = Convert.ToBase64String(audio), AudioFormat = new() { Type = mime == "audio/mpeg" ? "mp3" : mime[6..] } };
        yield return new SpeechFinishPart { ProviderMetadata = (response as SpeechProviderResponse)?.ProviderMetadata };
    }

    private async Task<HttpResponseMessage> SubmitAsync(string endpoint, JsonObject options, string accept,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        using var form = new MultipartFormDataContent();
        AddFields(form, options);
        using var message = new HttpRequestMessage(HttpMethod.Post, endpoint) { Content = form };
        NativeHttp.ApplyHeaders(message, headers);
        message.Headers.Authorization = new("Bearer", RequiredKey(credentials, Id));
        message.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue(accept));
        return await NativeHttp.SendAsync(client, message, ct);
    }

    private static JsonObject Options(Dictionary<string, JsonElement>? extras, params string[] additionallyReserved)
    {
        var reserved = new HashSet<string>(additionallyReserved, StringComparer.OrdinalIgnoreCase)
        { Id, "model", "prompt", "input", "text", "response_format", "stream", "authorization", "headers", "api-key", "apiKey", "api_key", "x-api-key", "url", "endpoint", "providerOptions" };
        var result = new JsonObject();
        if (extras?.TryGetValue(Id, out var native) == true)
        {
            if (native.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Stability AI options must be an object.");
            foreach (var field in native.EnumerateObject()) Add(field.Name, field.Value);
        }
        if (extras is not null)
            foreach (var (name, value) in extras) Add(name, value);
        return result;

        void Add(string name, JsonElement value)
        {
            if (reserved.Contains(name)) return;
            if (name.Length == 0 || name.Any(char.IsControl)) throw new ProviderRequestException("Invalid Stability AI option name.");
            // Known fields are canonicalized so case variants cannot conflict with typed fields.
            if (name.Equals("output_format", StringComparison.OrdinalIgnoreCase)) name = "output_format";
            if (name.Equals("aspect_ratio", StringComparison.OrdinalIgnoreCase)) name = "aspect_ratio";
            result[name] = Copy(value);
        }
    }

    private static string? Text(JsonObject options, string key) => options[key] is null ? null
        : options[key] is JsonValue value && value.TryGetValue<string>(out var text) ? text
        : throw new ProviderRequestException($"Stability AI option '{key}' must be a string.");

    private static string? ImageEndpoint(string model) => model.ToLowerInvariant() switch
    {
        "stable-image-core" => "stable-image/generate/core",
        "stable-image-ultra" => "stable-image/generate/ultra",
        var sd when sd.StartsWith("sd3", StringComparison.Ordinal) => "stable-image/generate/sd3",
        _ => null
    };

    private static string AspectRatio(string size)
    {
        var parts = size.ToLowerInvariant().Split('x');
        if (parts.Length != 2 || !int.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var width) || width <= 0
            || !int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var height) || height <= 0) return "1:1";
        return Ratios.MinBy(ratio => { var values = ratio.Split(':'); return Math.Abs((double)width / height - double.Parse(values[0], CultureInfo.InvariantCulture) / double.Parse(values[1], CultureInfo.InvariantCulture)); })!;
    }

    private static Dictionary<string, JsonElement> Metadata(HttpResponseMessage response)
        => new() { [Id] = JsonSerializer.SerializeToElement(response.Headers.Concat(response.Content.Headers)
            .Where(header => header.Key.ToLowerInvariant() is "seed" or "finish-reason" or "x-request-id" or "content-type")
            .ToDictionary(header => header.Key.ToLowerInvariant(), header => string.Join(", ", header.Value)), Json) };
}
