using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Deepdub;

public static class DeepdubRegistration
{
    public static IServiceCollection AddDeepdub(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, DeepdubProvider.Id, "https://restapi.deepdub.ai/");
        ProviderServiceCollectionExtensions.AddClient(services, DeepdubProvider.EuClient, "https://restapi.eu.deepdub.ai/");
        return services.AddProvider<DeepdubProvider>(DeepdubProvider.Id);
    }
}

internal sealed class DeepdubProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, ISpeechProvider, IStreamingSpeechEventsProvider, IChatAudioRoutingProvider
{
    internal const string Id = "deepdub";
    internal const string EuClient = "deepdub-eu";
    private const int AudioLimit = 32 * 1024 * 1024;
    private static readonly string[] Models = ["dd-etts-4.0", "dd-etts-3.0"];
    private sealed record SpeechPlan(JsonObject Payload, string Client, string MimeType, int? Rate);

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        await Task.CompletedTask;
        foreach (var model in Models)
        {
            ct.ThrowIfCancellationRequested();
            yield return new() { Id = model, Name = model, OwnedBy = "Deepdub", Type = "speech" };
        }
    }

    public bool IsSpeechModel(string model) => Models.Contains(model, StringComparer.Ordinal);
    public bool IsTranscriptionModel(string model) => false;

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var plan = Plan(request.Model, request.Input, request.Voice, null, request.ResponseFormat, request.Speed, null, OpenAIOptions(request));
        using var upstream = await OpenAsync(plan, headers, ct);
        await using var stream = await upstream.Content.ReadAsStreamAsync(ct);
        using var buffer = new MemoryStream();
        var chunk = new byte[16384];
        while (await stream.ReadAsync(chunk, ct) is > 0 and var length)
        {
            if (buffer.Length + length > AudioLimit) throw new IOException("Deepdub audio exceeds the gateway's inline size limit.");
            buffer.Write(chunk, 0, length);
        }
        if (buffer.Length == 0) throw new IOException("Deepdub returned empty audio.");
        var content = new ByteArrayContent(buffer.GetBuffer(), 0, (int)buffer.Length);
        content.Headers.ContentType = new(MimeType(upstream, plan));
        return new SpeechProviderResponse(content, Metadata(ResponseHeaders(upstream)));
    }

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamAsync(Plan(request.Model, request.Input, request.Voice, null, request.ResponseFormat, request.Speed, null, OpenAIOptions(request)),
            request.Model, request.Instructions, includeRaw: false, headers, ct);

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Text is not { Length: > 0 } || request.Text.Any(part => part is not { Type: "text", Text: not null }))
            throw new ProviderRequestException("Deepdub requires text parts.");
        var text = string.Concat(request.Text.Select(part => part.Text));
        var rate = request.OutputFormat?.Rate;
        if (rate is { } value && (!double.IsFinite(value) || value != Math.Truncate(value) || value is <= 0 or > int.MaxValue))
            throw new ProviderRequestException("Deepdub sample rate must be a positive integer.");

        SpeechPlan? plan = null;
        foreach (var part in request.Text.Where(part => part.Text.Length > 0))
        {
            var options = new JsonObject(new JsonNodeOptions { PropertyNameCaseInsensitive = true });
            AddSelected(options, request.ProviderOptions);
            AddSelected(options, part.ProviderOptions);
            var current = Plan(request.Model, text, part.Voice ?? request.Voice, request.Language, request.OutputFormat?.Type,
                request.Speed, (int?)rate, options);
            if (plan is not null && (plan.Client != current.Client || !JsonNode.DeepEquals(plan.Payload, current.Payload)))
                throw new ProviderRequestException("Deepdub requires one voice and configuration per request.", 422);
            plan = current;
        }
        return StreamAsync(plan ?? throw new ProviderRequestException("Deepdub text cannot be empty."), request.Model,
            request.Instructions, request.IncludeRawChunks == true, headers, ct);
    }

    private async IAsyncEnumerable<StreamingSpeechPart> StreamAsync(SpeechPlan plan, string model, string? instructions,
        bool includeRaw, IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var upstream = await OpenAsync(plan, headers, ct);
        await using var stream = await upstream.Content.ReadAsStreamAsync(ct);
        var buffer = new byte[16384];
        var first = await stream.ReadAsync(buffer, ct);
        if (first == 0) throw new IOException("Deepdub returned empty audio.");
        var responseHeaders = ResponseHeaders(upstream);
        yield return new SpeechStreamStartPart { Warnings = string.IsNullOrWhiteSpace(instructions)
            ? [] : [new { type = "unsupported", feature = "instructions" }] };
        yield return new SpeechResponseMetadataPart { Timestamp = DateTime.UtcNow, ModelId = model, Headers = responseHeaders };
        var mime = MimeType(upstream, plan);
        for (var length = first; length > 0; length = await stream.ReadAsync(buffer, ct))
        {
            var encoded = Convert.ToBase64String(buffer, 0, length);
            if (includeRaw) yield return new SpeechRawPart { RawValue = encoded };
            yield return new SpeechAudioPart { Audio = encoded, AudioFormat = new() { Type = mime, Rate = plan.Rate } };
        }
        yield return new SpeechFinishPart { ProviderMetadata = Metadata(responseHeaders) };
    }

    private async Task<HttpResponseMessage> OpenAsync(SpeechPlan plan, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(plan.Client);
        using var request = new HttpRequestMessage(HttpMethod.Post, "api/v1/tts") { Content = JsonContent.Create(plan.Payload, options: Json) };
        request.Headers.Add("x-api-key", RequiredKey(credentials, Id));
        NativeHttp.ApplyHeaders(request, headers);
        return await NativeHttp.SendAsync(client, request, ct);
    }

    private static SpeechPlan Plan(string model, string text, string? voice, string? language, string? format, double? speed, int? rate, JsonObject payload)
    {
        if (string.IsNullOrWhiteSpace(text)) throw new ProviderRequestException("Deepdub text cannot be empty.");
        var client = ReadString(payload, "region")?.Trim().ToLowerInvariant() switch
        {
            null or "us" => Id,
            "eu" => EuClient,
            _ => throw new ProviderRequestException("Deepdub region must be 'us' or 'eu'.")
        };
        payload.Remove("region");
        language = Nonempty(language) ?? Nonempty(ReadString(payload, "language")) ?? Nonempty(ReadString(payload, "locale"));
        payload.Remove("language");
        voice = Nonempty(voice) ?? Nonempty(ReadString(payload, "voicePromptId"));
        payload["model"] = model;
        payload["targetText"] = text;
        payload["locale"] = language ?? throw new ProviderRequestException("Deepdub requires a language or native locale.");
        payload["voicePromptId"] = voice ?? throw new ProviderRequestException("Deepdub requires a voice or native voicePromptId.");

        var (nativeFormat, mimeType) = (format ?? ReadString(payload, "format") ?? "mp3").Trim().ToLowerInvariant() switch
        {
            "mp3" or "mpeg" or "audio/mpeg" or "audio/mp3" => ("mp3", "audio/mpeg"),
            "opus" or "audio/opus" or "audio/ogg" => ("opus", "audio/ogg"),
            "mulaw" or "mu-law" or "ulaw" or "pcmu" or "audio/basic" or "audio/pcmu" => ("mulaw", "audio/basic"),
            _ => throw new ProviderRequestException("Deepdub supports mp3, opus and mulaw only.", 422)
        };
        payload["format"] = nativeFormat;
        if (speed is { } tempo) payload["tempo"] = tempo;
        if (rate is { } sampleRate) payload["sampleRate"] = sampleRate;
        else if (payload["sampleRate"] is { } configured)
            rate = configured is JsonValue value && value.TryGetValue<int>(out var parsed) && parsed > 0
                ? parsed : throw new ProviderRequestException("Deepdub sample rate must be a positive integer.");
        return new(payload, client, mimeType, rate ?? (nativeFormat == "mulaw" ? 8000 : null));
    }

    private static JsonObject OpenAIOptions(AudioSpeechRequest request)
    {
        var options = new JsonObject(new JsonNodeOptions { PropertyNameCaseInsensitive = true });
        if (request.AdditionalProperties is not { } extra) return options;
        AddOptions(options, extra);
        foreach (var name in new[] { "provider_options", "providerOptions" })
            if (Property(extra.GetValueOrDefault(name), Id) is { ValueKind: not JsonValueKind.Undefined } nested) AddObject(options, nested);
        if (extra.TryGetValue(Id, out var direct)) AddObject(options, direct);
        return options;
    }

    private static void AddSelected(JsonObject options, Dictionary<string, JsonElement>? selected)
    {
        if (selected?.TryGetValue(Id, out var native) == true) AddObject(options, native);
    }

    private static void AddObject(JsonObject options, JsonElement native)
    {
        if (native.ValueKind == JsonValueKind.Null) return;
        if (native.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Deepdub options must be a JSON object.");
        AddOptions(options, native.EnumerateObject().Select(p => KeyValuePair.Create(p.Name, p.Value)));
    }

    private static void AddOptions(JsonObject target, IEnumerable<KeyValuePair<string, JsonElement>> values)
    {
        foreach (var (name, value) in values)
            if (name.ToLowerInvariant() is not ("model" or "input" or "targettext" or "voice" or "response_format" or "stream_format"
                or "stream" or "instructions" or "speed" or "provider_options" or "provideroptions" or Id)) target[name] = Copy(value);
    }

    private static string? ReadString(JsonObject options, string name) => options[name] switch
    {
        null => null,
        JsonValue value when value.TryGetValue<string>(out var text) => text,
        _ => throw new ProviderRequestException($"Deepdub '{name}' must be a string.")
    };

    private static string? Nonempty(string? value) => string.IsNullOrWhiteSpace(value) ? null : value;
    private static string MimeType(HttpResponseMessage response, SpeechPlan plan)
        => response.Content.Headers.ContentType?.MediaType is { } type && type.StartsWith("audio/", StringComparison.OrdinalIgnoreCase) ? type : plan.MimeType;

    private static Dictionary<string, string> ResponseHeaders(HttpResponseMessage response)
        => NativeHttp.SafeResponseMetadataHeaders(response.Headers.Concat(response.Content.Headers)
            .ToDictionary(pair => pair.Key, pair => (string?)string.Join(", ", pair.Value), StringComparer.OrdinalIgnoreCase))
            .ToDictionary(pair => pair.Key, pair => pair.Value!, StringComparer.OrdinalIgnoreCase);

    private static Dictionary<string, JsonElement> Metadata(Dictionary<string, string> headers)
        => new() { [Id] = JsonSerializer.SerializeToElement(new { headers }, Json) };
}
