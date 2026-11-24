using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class AsticaProviderRegistration
{
    public static IServiceCollection AddAstica(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AsticaProvider.Id, "https://voice.astica.ai/");
        ProviderServiceCollectionExtensions.AddClient(services, AsticaProvider.DesignClient, "https://design.astica.ai/");
        return services.AddProvider<AsticaProvider>(AsticaProvider.Id);
    }
}

internal sealed class AsticaProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IImageModelRoutingProvider, ISpeechProvider, IStreamingSpeechEventsProvider, IChatAudioRoutingProvider
{
    internal const string Id = "astica";
    internal const string DesignClient = "astica-design";
    private const int MediaLimit = 25 * 1024 * 1024;
    private static readonly string[] ImageVersions = ["1.0_full", "2.0_full"];

    public Task<bool> IsImageModelAsync(string model, CancellationToken ct) => Task.FromResult(IsImageModel(model));
    public bool IsSpeechModel(string model) => !IsImageModel(model);
    public bool IsTranscriptionModel(string model) => false;

    private static bool IsImageModel(string model) => ImageVersions.Contains(model, StringComparer.OrdinalIgnoreCase);

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (credentials.GetApiKey(Id) is not { Length: > 0 } key || string.IsNullOrWhiteSpace(key)) yield break;
        foreach (var model in ImageVersions)
            yield return new() { Id = model, Name = model, OwnedBy = "Astica", Type = "image",
                Description = $"Astica Design image generation model version {model}." };

        using var client = clients.CreateClient(Id);
        var raw = await PostAsync(client, "api/voice_list", new() { ["tkn"] = key }, new Dictionary<string, string>(), ct);
        var voices = raw.ValueKind == JsonValueKind.Array ? raw : new[] { "voices", "data", "items", "results" }
            .Select(field => Property(raw, field)).FirstOrDefault(field => field.ValueKind == JsonValueKind.Array);
        var seen = new HashSet<string>(ImageVersions, StringComparer.OrdinalIgnoreCase);
        foreach (var voice in Array(voices))
        {
            var id = (String(voice, "voice") ?? String(voice, "voice_id") ?? String(voice, "voiceId") ?? String(voice, "id"))?.Trim();
            if (string.IsNullOrWhiteSpace(id) || !seen.Add(id)) continue;
            var tags = new[] { String(voice, "gender"), String(voice, "age") }.OfType<string>().Concat(
                new[] { "use_cases", "useCases", "usecases" }.SelectMany(field => Array(Property(voice, field)))
                    .Where(x => x.ValueKind == JsonValueKind.String).Select(x => x.GetString()!));
            yield return new() { Id = id, Name = String(voice, "name") ?? id, OwnedBy = "Astica", Type = "speech",
                Description = String(voice, "desc") ?? String(voice, "description"), Tags = tags.Where(x => !string.IsNullOrWhiteSpace(x)).Distinct().ToArray() };
        }
    }

    public async Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Prompt)) throw new ProviderRequestException("Astica image generation requires a prompt.");
        if (request.N is <= 0) throw new ProviderRequestException("n must be positive.");
        var options = Options(request.AdditionalProperties, "model", "modelVersion", "prompt", "n", "size", "aspect_ratio", "stream",
            "response_format", "partial_images", "images", "files", "mask");
        options["tkn"] = RequiredKey(credentials, Id);
        options["modelVersion"] = request.Model;
        options["prompt"] = request.Prompt;
        if (options["generate_quality"] is JsonValue quality && quality.TryGetValue<string>(out var text))
        {
            var value = text.Trim().ToLowerInvariant();
            if (value is not ("high" or "standard" or "fast" or "faster"))
                throw new ProviderRequestException("Invalid Astica generate_quality.");
            options["generate_quality"] = value;
        }
        using var client = clients.CreateClient(DesignClient);
        var raw = await PostAsync(client, "generate_image", options, headers, ct);
        if (!Uri.TryCreate(String(raw, "output"), UriKind.Absolute, out var target) || target.Scheme != "https"
            || target.UserInfo.Length != 0 || !target.IsDefaultPort
            || !(target.IdnHost == "astica.ai" || target.IdnHost.EndsWith(".astica.ai", StringComparison.OrdinalIgnoreCase)))
            throw new JsonException("Astica returned an invalid image URL.");

        // Output downloads are credential-free, with redirects disabled by the named client.
        using var download = new HttpRequestMessage(HttpMethod.Get, target);
        using var response = await NativeHttp.SendAsync(client, download, ct);
        var format = response.Content.Headers.ContentType?.MediaType switch
        {
            "image/png" => "png", "image/jpeg" => "jpeg", "image/webp" => "webp", "image/gif" => "gif",
            "image/bmp" => "bmp", "image/avif" => "avif",
            null or "application/octet-stream" => Path.GetExtension(target.AbsolutePath).ToLowerInvariant() switch
            { ".png" => "png", ".webp" => "webp", ".gif" => "gif", ".bmp" => "bmp", ".jpg" or ".jpeg" => "jpeg", _ => throw new JsonException("Unknown Astica image format.") },
            _ => throw new JsonException("Astica did not return a supported image content type.")
        };
        await using var content = await response.Content.ReadAsStreamAsync(ct);
        var image = await BoundedBase64Async(content, MediaLimit, ct);
        var extensions = new Dictionary<string, JsonElement> { ["providerMetadata"] = JsonSerializer.SerializeToElement(Metadata(raw), Json) };
        List<object> warnings = [];
        if (request.N is > 1) warnings.Add(Warning("n", "Astica generates one image per request."));
        if (request.Size is not (null or "1024x1024" or "auto")) warnings.Add(Warning("size", "Astica chooses image dimensions."));
        if (request.AdditionalProperties?.ContainsKey("aspect_ratio") == true) warnings.Add(Warning("aspectRatio", "Astica chooses the aspect ratio."));
        if (warnings.Count > 0) extensions["warnings"] = JsonSerializer.SerializeToElement(warnings, Json);
        return new() { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), OutputFormat = format, Data = [new() { B64Json = image }],
            AdditionalProperties = extensions };
    }

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => throw new ProviderRequestException("Astica Design does not support image editing.", 422);

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        var raw = await PostAsync(client, "api/tts", SpeechBody(request.Model, request.Input, request.AdditionalProperties, stream: false), headers, ct);
        var encoded = String(raw, "audio_b64");
        if (string.IsNullOrWhiteSpace(encoded) || encoded.Length > MediaLimit * 4 / 3 + 4)
            throw new JsonException("Astica returned missing or oversized audio.");
        byte[] bytes;
        try { bytes = Convert.FromBase64String(encoded); }
        catch (FormatException) { throw new JsonException("Astica returned invalid base64 audio."); }
        if (bytes.Length == 0 || bytes.Length > MediaLimit) throw new JsonException("Astica returned missing or oversized audio.");
        var audio = new ByteArrayContent(bytes);
        audio.Headers.ContentType = new("audio/wav");
        var metadata = ToObject(raw);
        metadata.Remove("audio_b64");
        return new SpeechProviderResponse(audio, Metadata(JsonSerializer.SerializeToElement(metadata, Json)));
    }

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamAsync(SpeechBody(request.Model, request.Input, request.AdditionalProperties, stream: true), request.Model, false, [], headers, ct);

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Text is not { Length: > 0 } || request.Text.Any(part => part?.Type != "text" || part.Text is null))
            throw new ProviderRequestException("Astica speech requires text parts.");
        List<object> warnings = [];
        if ((request.Voice is not null && request.Voice != request.Model) || request.Text.Any(part => part.Voice is not null && part.Voice != request.Model))
            warnings.Add(Warning("voice", "The voice is selected by the model ID."));
        if (request.Text.Any(part => part.ProviderOptions?.Count > 0)) warnings.Add(Warning("providerOptions", "Per-part options are not supported."));
        if (request.OutputFormat is not null) warnings.Add(Warning("outputFormat", "The voice engine chooses the format."));
        if (request.Language is not null) warnings.Add(Warning("language", "The voice engine chooses the language."));
        if (request.Speed is not null) warnings.Add(Warning("speed", "The voice engine chooses the speed."));
        if (request.Instructions is not null) warnings.Add(Warning("instructions", "Instructions are not supported."));
        var options = request.ProviderOptions?.GetValueOrDefault(Id) ?? default;
        if (options.ValueKind is not (JsonValueKind.Undefined or JsonValueKind.Null or JsonValueKind.Object))
            throw new ProviderRequestException("Astica options must be an object.");
        var text = string.Concat(request.Text.Select(part => part.Text));
        return StreamAsync(SpeechBody(request.Model, text, ReadOptions(options), stream: true), request.Model,
            request.IncludeRawChunks == true, warnings, headers, ct);
    }

    private async IAsyncEnumerable<StreamingSpeechPart> StreamAsync(JsonObject payload, string model, bool raw, List<object> warnings,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        using var request = new HttpRequestMessage(HttpMethod.Post, "api/tts") { Content = JsonContent.Create(payload, options: Json) };
        NativeHttp.ApplyHeaders(request, headers);
        using var response = await NativeHttp.SendAsync(client, request, ct);
        var type = response.Content.Headers.ContentType?.MediaType;
        if (type?.StartsWith("audio/", StringComparison.OrdinalIgnoreCase) != true && type != "application/octet-stream")
            throw new JsonException("Astica did not return an audio stream.");
        await using var source = await response.Content.ReadAsStreamAsync(ct);
        var buffer = new byte[16384];
        var read = await source.ReadAsync(buffer, ct);
        if (read == 0) throw new JsonException("Astica returned an empty audio stream.");
        yield return new SpeechStreamStartPart { Warnings = warnings };
        yield return new SpeechResponseMetadataPart { ModelId = model, Timestamp = DateTime.UtcNow };
        do
        {
            var chunk = Convert.ToBase64String(buffer, 0, read);
            if (raw) yield return new SpeechRawPart { RawValue = chunk };
            yield return new SpeechAudioPart { Audio = chunk, AudioFormat = new() { Type = type! } };
        } while ((read = await source.ReadAsync(buffer, ct)) > 0);
        yield return new SpeechFinishPart();
    }

    private JsonObject SpeechBody(string model, string input, IEnumerable<KeyValuePair<string, JsonElement>>? source, bool stream)
    {
        if (string.IsNullOrWhiteSpace(input)) throw new ProviderRequestException("Astica speech requires text.");
        var body = Options(source, "model", "input", "text", "voice", "stream", "stream_format", "response_format", "language");
        body["tkn"] = RequiredKey(credentials, Id);
        body["text"] = input;
        body["voice"] = model;
        body["stream"] = stream;
        return body;
    }

    private static async Task<JsonElement> PostAsync(HttpClient client, string path, JsonObject body, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var result = await NativeHttp.PostAsync<JsonObject, JsonElement>(client, path, body, _ => { }, headers, Json, ct);
        if (String(result, "status")?.Equals("error", StringComparison.OrdinalIgnoreCase) == true)
            throw new ProviderRequestException("Astica rejected the request.", 502);
        return result;
    }

    private static Dictionary<string, JsonElement> Metadata(JsonElement value) => new() { [Id] = value };
    private static object Warning(string feature, string reason) => new { type = "ignored", feature, reason };

    private static IEnumerable<KeyValuePair<string, JsonElement>> ReadOptions(JsonElement source)
        => source.ValueKind == JsonValueKind.Object ? source.EnumerateObject().Select(x => KeyValuePair.Create(x.Name, x.Value)) : [];

    private static JsonObject Options(IEnumerable<KeyValuePair<string, JsonElement>>? source, params string[] fields)
    {
        var reserved = new HashSet<string>(fields, StringComparer.OrdinalIgnoreCase)
            { Id, "tkn", "authorization", "api_key", "apiKey", "x-api-key", "headers" };
        var result = new JsonObject();
        var options = source?.ToArray() ?? [];
        var scoped = options.FirstOrDefault(x => x.Key == Id).Value;
        if (scoped.ValueKind is not (JsonValueKind.Undefined or JsonValueKind.Null or JsonValueKind.Object))
            throw new ProviderRequestException("Astica options must be an object.");
        foreach (var (field, value) in ReadOptions(scoped).Concat(options))
            if (!reserved.Contains(field) && value.ValueKind != JsonValueKind.Undefined) result[field] = Copy(value);
        return result;
    }
}
