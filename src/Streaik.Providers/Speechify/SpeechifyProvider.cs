using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class SpeechifyProviderRegistration
{
    public static IServiceCollection AddSpeechify(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, SpeechifyProvider.Id, "https://api.sws.speechify.com/");
        ProviderServiceCollectionExtensions.AddClient(services, SpeechifyProvider.SdkStreamClient, "https://api.speechify.ai/");
        return services.AddProvider<SpeechifyProvider>(SpeechifyProvider.Id);
    }
}

internal sealed partial class SpeechifyProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IStreamingSpeechProvider, IStreamingSpeechEventsProvider, IChatAudioRoutingProvider
{
    internal const string Id = "speechify";
    internal const string SdkStreamClient = "speechify-stream";
    private const int MaxBase64Length = ((25 * 1024 * 1024 + 2) / 3) * 4;

    public bool IsSpeechModel(string model) => !string.IsNullOrWhiteSpace(model);
    public bool IsTranscriptionModel(string model) => false;

    public Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeakBufferedAsync(Plan(request, stream: false), headers, ct);

    public Task<HttpResponseMessage> SpeakStreamingAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var plan = Plan(request, stream: true);
        // Speechify only serves WAV from the buffered endpoint.
        return plan.Format == "wav" ? SpeakBufferedAsync(Plan(request, stream: false), headers, ct)
            : SendStreamAsync(plan, Id, headers, ct);
    }

    public async IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var response = await SpeakStreamingAsync(request, headers, ct);
        await foreach (var part in ReadAudioAsync(response, request.Model, request.Instructions, request.Speed is not null, raw: false, rate: null, ct))
            yield return part;
    }

    public async IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        if (request.Text is not { Length: > 0 } || request.Text.Any(part => part?.Text is null))
            throw new ProviderRequestException("Speechify requires text parts.");
        SpeechPlan? plan = null;
        foreach (var part in request.Text)
        {
            var options = new Dictionary<string, JsonElement>(StringComparer.OrdinalIgnoreCase);
            Merge(options, request.ProviderOptions);
            Merge(options, part.ProviderOptions);
            var next = Plan(new AudioSpeechRequest { Model = request.Model, Input = "", Voice = part.Voice ?? request.Voice,
                ResponseFormat = request.OutputFormat?.Type, AdditionalProperties = options }, stream: true);
            if (plan is not null && !JsonNode.DeepEquals(plan.Body, next.Body))
                throw new ProviderRequestException("Speechify streaming requires one effective voice and options configuration across text parts.");
            plan = next;
        }
        if (plan!.Format == "wav") throw new ProviderRequestException("Speechify does not support WAV on its streaming endpoint.");
        if (request.OutputFormat?.Rate is { } rate && rate != SampleRate(plan.Format))
            throw new ProviderRequestException("Speechify outputFormat.rate must match the selected native output format.");
        plan.Body["input"] = string.Concat(request.Text.Select(part => part.Text));
        using var response = await SendStreamAsync(plan, SdkStreamClient, headers, ct);
        await foreach (var part in ReadAudioAsync(response, request.Model, request.Instructions, request.Speed is not null,
            request.IncludeRawChunks == true, SampleRate(plan.Format), ct)) yield return part;
    }

    private async Task<HttpResponseMessage> SpeakBufferedAsync(SpeechPlan plan, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        NeedText(plan);
        using var client = clients.CreateClient(Id);
        var (raw, upstream) = await NativeHttp.PostJsonAsync(client, "v1/audio/speech", plan.Body, Bearer(), headers, Json, ct);
        if (String(raw, "audio_data") is not { Length: > 0 } audio || audio.Length > MaxBase64Length)
            throw new JsonException("Speechify returned missing or oversized speech audio.");
        byte[] bytes;
        try { bytes = Convert.FromBase64String(audio); }
        catch (FormatException ex) { throw new JsonException("Speechify returned invalid base64 audio.", ex); }
        if (bytes.Length == 0) throw new JsonException("Speechify returned empty speech audio.");

        var content = new ByteArrayContent(bytes);
        content.Headers.ContentType = new(Mime(Normalize(String(raw, "audio_format")) ?? plan.Format));
        var metadata = ToObject(raw);
        metadata.Remove("audio_data");
        var safe = NativeHttp.SafeResponseMetadataHeaders(upstream);
        if (safe.Count > 0 && !metadata.ContainsKey("headers")) metadata["headers"] = JsonSerializer.SerializeToNode(safe, Json);
        return new SpeechProviderResponse(content, Metadata(metadata));
    }

    private async Task<HttpResponseMessage> SendStreamAsync(SpeechPlan plan, string clientName, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        NeedText(plan);
        using var client = clients.CreateClient(clientName);
        using var request = new HttpRequestMessage(HttpMethod.Post, "v1/audio/stream") { Content = JsonContent.Create(plan.Body, options: Json) };
        Bearer()(request.Headers);
        NativeHttp.ApplyHeaders(request, headers);
        request.Headers.Accept.ParseAdd(Mime(plan.Format));
        var response = await NativeHttp.SendAsync(client, request, ct);
        var type = response.Content.Headers.ContentType?.MediaType;
        if (type is null or "application/octet-stream") response.Content.Headers.ContentType = new(Mime(plan.Format));
        else if (!type.StartsWith("audio/", StringComparison.OrdinalIgnoreCase))
        {
            response.Dispose();
            throw new JsonException("Speechify did not return a binary audio stream.");
        }
        return response;
    }

    private static async IAsyncEnumerable<StreamingSpeechPart> ReadAudioAsync(HttpResponseMessage response, string model,
        string? instructions, bool speed, bool raw, int? rate, [EnumeratorCancellation] CancellationToken ct)
    {
        var contentType = response.Content.Headers.ContentType;
        if (int.TryParse(contentType?.Parameters.FirstOrDefault(p => p.Name.Equals("rate", StringComparison.OrdinalIgnoreCase))?.Value?.Trim('"'), out var actualRate)
            && actualRate > 0) rate = actualRate;
        var headers = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, values) in response.Headers.Concat(response.Content.Headers)) headers[name] = string.Join(", ", values);
        var warnings = new List<object>();
        if (!string.IsNullOrWhiteSpace(instructions)) warnings.Add(new { type = "unsupported", feature = "instructions" });
        if (speed) warnings.Add(new { type = "unsupported", feature = "speed" });
        await using var audio = await response.Content.ReadAsStreamAsync(ct);
        var buffer = new byte[16 * 1024];
        var count = await audio.ReadAsync(buffer, ct);
        if (count == 0) throw new JsonException("Speechify returned empty speech audio.");
        yield return new SpeechStreamStartPart { Warnings = warnings };
        yield return new SpeechResponseMetadataPart { ModelId = model, Timestamp = DateTime.UtcNow,
            Headers = NativeHttp.SafeResponseMetadataHeaders(headers).ToDictionary(pair => pair.Key, pair => pair.Value!) };
        do
        {
            var encoded = Convert.ToBase64String(buffer, 0, count);
            if (raw) yield return new SpeechRawPart { RawValue = encoded };
            yield return new SpeechAudioPart { Audio = encoded, AudioFormat = new() { Type = contentType?.MediaType ?? "application/octet-stream", Rate = rate } };
        } while ((count = await audio.ReadAsync(buffer, ct)) > 0);
        yield return new SpeechFinishPart();
    }

    private static SpeechPlan Plan(AudioSpeechRequest request, bool stream)
    {
        var model = request.Model.Split('/', StringSplitOptions.TrimEntries);
        if (model.Length is < 1 or > 2 || model.Any(string.IsNullOrWhiteSpace))
            throw new ProviderRequestException("Speechify model must be '{model}' or '{model}/{voiceId}'.");
        var options = request.AdditionalProperties is null ? new(StringComparer.OrdinalIgnoreCase)
            : new Dictionary<string, JsonElement>(request.AdditionalProperties, StringComparer.OrdinalIgnoreCase);
        var suppliedVoice = TakeString(options, "voice_id");
        var voice = (model.Length == 2 ? model[1] : suppliedVoice ?? request.Voice)?.Trim();
        if (string.IsNullOrWhiteSpace(voice)) throw new ProviderRequestException("Speechify requires a voice_id or a model/voice shortcut.");
        var audioFormat = TakeString(options, "audio_format");
        var outputFormat = TakeString(options, "output_format");
        var format = Normalize(request.ResponseFormat ?? (stream ? outputFormat ?? audioFormat : audioFormat)) ?? (stream ? "mp3" : "wav");
        if (stream) format = StreamFormat(format);
        var body = new JsonObject();
        foreach (var (name, value) in options)
        {
            if (name.ToLowerInvariant() is "model" or "input" or "voice" or "response_format" or "stream_format" or "stream") continue;
            if (name.Equals("options", StringComparison.OrdinalIgnoreCase))
            {
                if (value.ValueKind == JsonValueKind.Null) continue;
                if (value.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Speechify options must be an object.");
                body["options"] = Copy(value);
            }
            else body[name] = Copy(value);
        }
        body["model"] = model[0];
        body["voice_id"] = voice;
        body["input"] = request.Input;
        body[stream ? "output_format" : "audio_format"] = format;
        return new(body, format);
    }

    private static string? TakeString(Dictionary<string, JsonElement> options, string key)
        => !options.Remove(key, out var value) || value.ValueKind == JsonValueKind.Null ? null
            : value.ValueKind == JsonValueKind.String ? value.GetString() : throw new ProviderRequestException($"Speechify '{key}' must be a string.");

    private static void Merge(Dictionary<string, JsonElement> target, Dictionary<string, JsonElement>? all)
    {
        if (all?.FirstOrDefault(p => p.Key.Equals(Id, StringComparison.OrdinalIgnoreCase)).Value is not { } value || value.ValueKind == JsonValueKind.Undefined) return;
        if (value.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("providerOptions.speechify must be an object.");
        foreach (var property in value.EnumerateObject()) target[property.Name] = property.Value;
    }

    private Action<HttpRequestHeaders> Bearer() => headers => headers.Authorization = new("Bearer", RequiredKey(credentials, Id));
    private static Dictionary<string, JsonElement> Metadata(JsonObject raw) => new() { [Id] = JsonSerializer.SerializeToElement(raw, Json) };
    private static void NeedText(SpeechPlan plan)
    {
        if (string.IsNullOrWhiteSpace(plan.Body["input"]?.GetValue<string>())) throw new ProviderRequestException("Speechify requires nonempty speech input.");
    }

    private static string? Normalize(string? value) => value?.Trim().ToLowerInvariant() switch
    {
        null or "" => null, "wave" or "audio/wav" or "audio/x-wav" => "wav", "mpeg" or "audio/mpeg" => "mp3",
        "audio/aac" => "aac", "audio/ogg" or "audio/opus" or "opus" => "ogg", "audio/pcm" or "audio/l16" or "pcm16" or "raw" => "pcm",
        "audio/basic" or "audio/ulaw" or "mulaw" or "mu-law" or "u-law" => "ulaw", var format => format
    };

    private static string StreamFormat(string value) => value switch
    {
        "mp3" => "mp3_24000_128", "aac" => "aac_24000", "ogg" => "ogg_24000", "pcm" => "pcm_24000", "ulaw" => "ulaw_8000",
        "wav" or "ogg_24000" or "aac_24000" or "ulaw_8000" => value,
        _ when value.StartsWith("mp3_", StringComparison.Ordinal) || value.StartsWith("pcm_", StringComparison.Ordinal) => value,
        _ => throw new ProviderRequestException("Unsupported Speechify streaming output format.")
    };

    private static string Mime(string format) => format.Split('_', 2)[0] switch
    {
        "mp3" => "audio/mpeg", "aac" => "audio/aac", "ogg" => "audio/ogg", "wav" => "audio/wav", "pcm" => "audio/L16",
        "ulaw" => "audio/basic", _ => "application/octet-stream"
    };

    private static int? SampleRate(string format) => format.Split('_').Skip(1).FirstOrDefault() is { } part
        && int.TryParse(part, out var rate) && rate >= 8000 ? rate : null;
    private sealed record SpeechPlan(JsonObject Body, string Format);
}
