using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class UnrealSpeechProviderRegistration
{
    public static IServiceCollection AddUnrealSpeech(this IServiceCollection services)
    {
        const string address = "https://api.v8.unrealspeech.com/";
        ProviderServiceCollectionExtensions.AddClient(services, UnrealSpeechProvider.Id, address);
        ProviderServiceCollectionExtensions.AddClient(services, UnrealSpeechProvider.AssetClient, address);
        return services.AddProvider<UnrealSpeechProvider>(UnrealSpeechProvider.Id);
    }
}

internal sealed class UnrealSpeechProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IStreamingSpeechProvider, IChatAudioRoutingProvider
{
    internal const string Id = "unrealspeech";
    internal const string AssetClient = "unrealspeech-assets";

    private static readonly string[] Voices = ["Autumn", "Melody", "Hannah", "Emily", "Ivy", "Kaitlyn", "Luna", "Willow", "Lauren", "Sierra",
        "Noah", "Jasper", "Caleb", "Ronan", "Ethan", "Daniel", "Zane", "Mei", "Lian", "Ting", "Jing", "Wei", "Jian", "Hao", "Sheng",
        "Lucía", "Mateo", "Javier", "Élodie", "Ananya", "Priya", "Arjun", "Rohan", "Giulia", "Luca", "Camila", "Thiago", "Rafael"];
    private static readonly HashSet<string> Reserved = new(StringComparer.OrdinalIgnoreCase)
    {
        "text", "voiceId", "outputFormat", "model", "voice", "input", "response_format", "stream_format", "stream",
        "instructions", "language", "providerOptions", "providerMetadata", "authorization", "api_key", "apiKey", "x-api-key", "headers"
    };
    private static readonly HashSet<string> Mapped = new(StringComparer.OrdinalIgnoreCase)
        { "bitrate", "speed", "pitch", "timestampType", "codec", "temperature" };

    public bool IsSpeechModel(string model) => Voices.Contains(model, StringComparer.OrdinalIgnoreCase);
    public bool IsTranscriptionModel(string model) => false;

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        foreach (var voice in Voices)
        {
            ct.ThrowIfCancellationRequested();
            yield return new Model { Id = voice, Name = voice, OwnedBy = "UnrealSpeech", Type = "speech", Description = $"UnrealSpeech TTS voice {voice}." };
        }
    }

    public Task<HttpResponseMessage> SpeakStreamingAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeakAsync(request, headers, ct);

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Input)) throw new ProviderRequestException("UnrealSpeech requires text input.");
        var voice = Voices.FirstOrDefault(voice => voice.Equals(request.Model, StringComparison.OrdinalIgnoreCase)) ?? request.Model;
        var charCount = request.Input.Length;
        var endpoint = charCount <= 1000 ? "stream" : charCount <= 3000 ? "speech" : "synthesisTasks";
        var options = Options(request.AdditionalProperties);
        var bitrate = Option<string>(options, "bitrate")?.Trim().ToLowerInvariant() ?? "192k";
        if (bitrate is not ("16k" or "32k" or "48k" or "64k" or "128k" or "192k" or "256k" or "320k"))
            throw new ProviderRequestException("Unsupported UnrealSpeech bitrate.");
        var speed = Range(Option<double?>(options, "speed") ?? request.Speed ?? 0, -1, 1, "speed");
        var pitch = Range(Option<double?>(options, "pitch") ?? 1, 0.5, 1.5, "pitch");
        var timestamp = Option<string>(options, "timestampType")?.Trim().ToLowerInvariant();
        if (string.IsNullOrEmpty(timestamp)) timestamp = "sentence";
        if (timestamp is not ("word" or "sentence")) throw new ProviderRequestException("UnrealSpeech timestampType must be word or sentence.");
        var codec = Option<string>(options, "codec")?.Trim().ToLowerInvariant();
        if (string.IsNullOrEmpty(codec)) codec = request.ResponseFormat?.ToLowerInvariant() switch
        {
            "pcm" or "pcm16" or "raw" => "pcm_s16le", "mulaw" or "ulaw" => "pcm_mulaw", _ => null
        };
        if (codec is not (null or "libmp3lame" or "pcm_mulaw" or "pcm_s16le")) throw new ProviderRequestException("Unsupported UnrealSpeech codec.");
        var temperature = Option<double?>(options, "temperature");
        if (temperature is { } value) Range(value, 0.1, 0.8, "temperature");

        var body = new Dictionary<string, object?> { ["Text"] = request.Input, ["VoiceId"] = voice,
            ["Bitrate"] = bitrate, ["Speed"] = speed, ["Pitch"] = pitch };
        foreach (var (name, option) in options)
            if (!Reserved.Contains(name) && !Mapped.Contains(name)) body[name] = option;
        if (endpoint == "stream")
        {
            if (codec is not null) body["Codec"] = codec;
            if (temperature is not null) body["Temperature"] = temperature;
        }
        else
        {
            body["OutputFormat"] = "uri";
            body["TimestampType"] = timestamp;
        }

        var key = RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        var metadata = new JsonObject { ["endpoint"] = endpoint, ["voiceId"] = voice, ["charCount"] = charCount,
            ["bitrate"] = bitrate, ["speed"] = speed, ["pitch"] = pitch, ["timestampType"] = timestamp,
            ["codec"] = codec, ["temperature"] = temperature };
        if (endpoint == "stream")
        {
            using var shortAudio = await SendAsync(client, HttpMethod.Post, endpoint, JsonContent.Create(body, options: Json), "Bearer", key, headers, ct);
            return WrapAudio(shortAudio, metadata, codec switch { "pcm_s16le" => "application/octet-stream", "pcm_mulaw" => "audio/basic", _ => "audio/mpeg" });
        }

        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromMinutes(20));
        var token = deadline.Token;
        try
        {
            var start = await JsonAsync<JsonElement>(client, HttpMethod.Post, endpoint, JsonContent.Create(body, options: Json), "Bearer", key, headers, token);
            var completed = start;
            var result = new JsonObject { ["endpoint"] = endpoint };
            if (endpoint == "synthesisTasks")
            {
                var taskId = TaskString(start, "TaskId");
                if (string.IsNullOrWhiteSpace(taskId) || taskId is "." or "..") throw new JsonException("Missing UnrealSpeech task ID.");
                completed = await PollAsync(client, taskId, key, headers, token);
                result["taskId"] = taskId;
                result["create"] = SafeResult(start);
                result["final"] = SafeResult(completed);
            }
            else result["response"] = SafeResult(start);
            metadata["result"] = result;

            var url = FindOutput(completed);
            if (!TryAssetUri(url, out var uri)) throw new JsonException("UnrealSpeech did not return a trusted audio URL.");
            using var assets = clients.CreateClient(AssetClient);
            using var download = new HttpRequestMessage(HttpMethod.Get, uri);
            using var audio = await NativeHttp.SendAsync(assets, download, token);
            return WrapAudio(audio, metadata, "audio/mpeg");
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested && deadline.IsCancellationRequested)
        {
            throw new TimeoutException("UnrealSpeech synthesis timed out.");
        }
    }

    private static async Task<JsonElement> PollAsync(HttpClient client, string taskId, string key, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        while (true)
        {
            var result = await JsonAsync<JsonElement>(client, HttpMethod.Get, $"synthesisTasks/{Uri.EscapeDataString(taskId)}", null, "Bearer", key, headers, ct);
            var state = TaskString(result, "TaskStatus")?.ToLowerInvariant();
            if (state == "completed") return result;
            if (state is "failed" or "error" or "cancelled" or "canceled") throw new IOException("UnrealSpeech synthesis failed.");
            await Task.Delay(TimeSpan.FromSeconds(2), ct);
        }
    }

    private static SpeechProviderResponse WrapAudio(HttpResponseMessage upstream, JsonObject metadata, string fallback)
    {
        var content = upstream.Content;
        var mime = content.Headers.ContentType?.MediaType;
        if (mime is not (null or "application/octet-stream") && !mime.StartsWith("audio/", StringComparison.OrdinalIgnoreCase))
            throw new IOException("UnrealSpeech returned an unexpected audio content type.");
        if (content.Headers.ContentLength == 0) throw new IOException("UnrealSpeech returned no audio.");
        if (mime is null or "application/octet-stream") content.Headers.ContentType = new MediaTypeHeaderValue(fallback);
        var safeHeaders = NativeHttp.SafeResponseMetadataHeaders(upstream.Headers.Concat(content.Headers)
            .ToDictionary(pair => pair.Key, pair => (string?)string.Join(", ", pair.Value), StringComparer.OrdinalIgnoreCase));
        if (safeHeaders.Count > 0) metadata["responseHeaders"] = JsonSerializer.SerializeToNode(safeHeaders, Json);
        if (content.Headers.ContentLength is { } size) metadata["bytes"] = size;
        var wrapped = new SpeechProviderResponse(content, new() { [Id] = JsonSerializer.SerializeToElement(metadata, Json) });
        upstream.Content = null; // Transfer ownership of the response stream to the gateway.
        return wrapped;
    }

    private static bool TryAssetUri(string? value, out Uri? uri)
    {
        if (!Uri.TryCreate(value, UriKind.Absolute, out uri) || uri.Scheme != Uri.UriSchemeHttps || !uri.IsDefaultPort
            || uri.UserInfo.Length != 0 || uri.Fragment.Length != 0) return false;
        var host = uri.IdnHost;
        if (host == "unrealspeech.com" || host.EndsWith(".unrealspeech.com", StringComparison.Ordinal)) return true;
        // The API docs show both audio and timestamp files in this provider-owned bucket.
        return host is "unreal-expire-in-90-days.s3-us-west-2.amazonaws.com" or "unreal-expire-in-90-days.s3.us-west-2.amazonaws.com";
    }

    private static JsonNode? SafeResult(JsonElement result)
        => result.ValueKind switch
        {
            JsonValueKind.Object => new JsonObject(result.EnumerateObject().Where(p => !p.Name.Equals("OutputUri", StringComparison.OrdinalIgnoreCase))
                .Select(p => KeyValuePair.Create(p.Name, SafeResult(p.Value)))),
            JsonValueKind.Array => new JsonArray(result.EnumerateArray().Select(SafeResult).ToArray()),
            JsonValueKind.String when Uri.TryCreate(result.GetString(), UriKind.Absolute, out var url) && url.Scheme == Uri.UriSchemeHttps && url.Query.Length > 0 => null,
            _ => Copy(result)
        };

    private static string? TaskString(JsonElement body, string field)
    {
        var value = TaskField(body, field);
        return value.ValueKind switch { JsonValueKind.String => value.GetString(), JsonValueKind.Number => value.GetRawText(), _ => null };
    }

    private static string? FindOutput(JsonElement body)
    {
        var value = TaskField(body, "OutputUri");
        if (value.ValueKind == JsonValueKind.String) return value.GetString();
        return Array(value).FirstOrDefault(item => item.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(item.GetString())) is { ValueKind: JsonValueKind.String } url
            ? url.GetString() : null;
    }

    private static JsonElement TaskField(JsonElement body, string field)
        => Field(body, field) is { ValueKind: not (JsonValueKind.Undefined or JsonValueKind.Null) } value ? value : Field(Field(body, "SynthesisTask"), field);

    private static JsonElement Field(JsonElement node, string field)
    {
        if (node.ValueKind == JsonValueKind.Object)
            foreach (var property in node.EnumerateObject())
                if (property.Name.Equals(field, StringComparison.OrdinalIgnoreCase)) return property.Value;
        return default;
    }

    private static Dictionary<string, JsonElement> Options(Dictionary<string, JsonElement>? input)
    {
        var options = new Dictionary<string, JsonElement>(StringComparer.OrdinalIgnoreCase);
        if (input?.TryGetValue(Id, out var scoped) == true)
        {
            if (scoped.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("UnrealSpeech options must be an object.");
            foreach (var property in scoped.EnumerateObject()) options[property.Name] = property.Value;
        }
        foreach (var (name, value) in input ?? []) if (!name.Equals(Id, StringComparison.OrdinalIgnoreCase)) options[name] = value;
        return options;
    }

    private static T? Option<T>(Dictionary<string, JsonElement> options, string name)
    {
        if (!options.TryGetValue(name, out var value) || value.ValueKind == JsonValueKind.Null) return default;
        try { return value.Deserialize<T>(Json); }
        catch (JsonException) { throw new ProviderRequestException($"Invalid UnrealSpeech {name}."); }
    }

    private static double Range(double value, double min, double max, string name)
        => double.IsFinite(value) && value >= min && value <= max ? value : throw new ProviderRequestException($"UnrealSpeech {name} must be between {min} and {max}.");
}
