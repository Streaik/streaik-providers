using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class RimeRegistration
{
    public static IServiceCollection AddRime(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, "rime", "https://users.rime.ai/");
        return services.AddProvider<RimeProvider>("rime");
    }
}

internal sealed class RimeProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, ISpeechProvider, IStreamingSpeechEventsProvider, IChatAudioRoutingProvider
{
    private const string Id = "rime";
    private const int MediaLimit = 25 * 1024 * 1024;
    private static readonly HashSet<string> Reserved = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "modelId", "text", "input", "prompt", "lang", "speaker", "format", "output_format", "response_format", "stream", "stream_format",
        "authorization", "headers", "api-key", "api_key", "apiKey", "x-api-key", "url", "endpoint", "providerOptions", Id,
        "voice", "language", "samplingRate", "speed", "instructions", "speedAlpha", "noTextNormalization", "saveOovs",
        "timeScaleFactor", "customLexicon", "pauseBetweenBrackets", "inlineSpeedAlpha", "phonemizeBetweenBrackets"
    };

    public bool IsSpeechModel(string model) => model.Split('/', 2)[0].ToLowerInvariant() is "coda" or "mistv3";
    public bool IsTranscriptionModel(string model) => false;

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        ct.ThrowIfCancellationRequested();
        yield return new() { Id = "coda", OwnedBy = "Rime", Name = "coda", Type = "speech", Tags = ["model:coda", "tts", "base"],
            Description = "Conversational text-to-speech with a dedicated speech inference engine." };
        yield return new() { Id = "mistv3", OwnedBy = "Rime", Name = "mistv3", Type = "speech", Tags = ["model:mistv3", "tts", "base"],
            Description = "Low-latency text-to-speech with inline pronunciation control." };
        using var client = clients.CreateClient(Id);
        // Discovery is public; do not send the user's API key.
        var root = await NativeHttp.GetAsync<JsonElement>(client, "data/voices/all-v2.json", _ => { }, ct);
        if (root.ValueKind != JsonValueKind.Object) yield break;
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var model in root.EnumerateObject())
        {
            var baseId = model.Name.Trim().ToLowerInvariant();
            if (baseId is not ("coda" or "mistv3") || model.Value.ValueKind != JsonValueKind.Object) continue;
            foreach (var language in model.Value.EnumerateObject())
            foreach (var entry in Array(language.Value))
            {
                ct.ThrowIfCancellationRequested();
                if (entry.ValueKind != JsonValueKind.String || NonEmpty(entry.GetString()) is not { } voice || voice.Contains('/')) continue;
                var modelId = $"{baseId}/{voice}";
                if (seen.Add(modelId)) yield return new() { Id = modelId, OwnedBy = "Rime", Type = "speech", Name = $"{baseId} / {voice} ({language.Name})",
                    Tags = [$"model:{baseId}", $"voice:{voice}", $"language:{language.Name}", "tts"] };
            }
        }
    }

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var plan = PlanFor(request, Options(request.AdditionalProperties, flat: true));
        using var response = await SendSpeechAsync(plan, headers, ct);
        var (mime, format, rate) = ResponseFormat(response, plan);
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        using var audio = new MemoryStream();
        var buffer = new byte[16 * 1024];
        while (await stream.ReadAsync(buffer, ct) is > 0 and var read)
        {
            if (audio.Length + read > MediaLimit) throw new IOException("Rime audio output is too large.");
            audio.Write(buffer, 0, read);
        }
        ValidateAudio(format, audio.Length);
        var content = new ByteArrayContent(audio.GetBuffer(), 0, (int)audio.Length);
        content.Headers.ContentType = MediaTypeHeaderValue.Parse(mime);
        return new SpeechProviderResponse(content, Metadata(plan, mime, format, rate, audio.Length, ResponseHeaders(response, mime)));
    }

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(AudioSpeechRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => StreamAsync(PlanFor(request, Options(request.AdditionalProperties, flat: true)), request.Model, false, headers, ct);

    public IAsyncEnumerable<StreamingSpeechPart> StreamSpeechAsync(StreamingSpeechRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Text is not { Length: > 0 } || request.Text.Any(part => part is null || part.Type != "text" || part.Text is null))
            throw new ProviderRequestException("Rime requires non-null text parts.");
        var text = string.Concat(request.Text.Select(part => part.Text));
        var rate = request.OutputFormat?.Rate;
        if (rate is { } value && (!double.IsFinite(value) || value < 1 || value > int.MaxValue || Math.Truncate(value) != value))
            throw new ProviderRequestException("Rime sampling rate must be a positive integer.");
        Plan? plan = null;
        foreach (var part in request.Text.Where(part => part.Text.Length > 0))
        {
            var options = Options(request.ProviderOptions);
            foreach (var (name, option) in Options(part.ProviderOptions)) options[name] = option;
            var current = PlanFor(new() { Model = request.Model, Input = text, Voice = part.Voice ?? request.Voice,
                ResponseFormat = request.OutputFormat?.Type, Speed = request.Speed, Instructions = request.Instructions }, options, request.Language, (int?)rate);
            if (plan is not null && !JsonElement.DeepEquals(JsonSerializer.SerializeToElement(plan.Body, Json), JsonSerializer.SerializeToElement(current.Body, Json)))
                throw new ProviderRequestException("Rime requires one effective voice and provider configuration per script.", 422);
            plan ??= current;
        }
        return StreamAsync(plan ?? throw new ProviderRequestException("Rime speech text cannot be empty."),
            request.Model, request.IncludeRawChunks == true, headers, ct);
    }

    private async IAsyncEnumerable<StreamingSpeechPart> StreamAsync(Plan plan, string model, bool raw,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var response = await SendSpeechAsync(plan, headers, ct);
        var (mime, format, rate) = ResponseFormat(response, plan);
        var responseHeaders = ResponseHeaders(response, mime);
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        yield return new SpeechStreamStartPart { Warnings = plan.Warnings };
        yield return new SpeechResponseMetadataPart { ModelId = model, Timestamp = DateTime.UtcNow, Headers = responseHeaders };
        var buffer = new byte[16 * 1024 + 1];
        var pending = 0;
        long bytes = 0;
        while (await stream.ReadAsync(buffer.AsMemory(pending), ct) is > 0 and var read)
        {
            if (raw) yield return new SpeechRawPart { RawValue = Convert.ToBase64String(buffer, pending, read) };
            var available = pending + read;
            var length = format == "pcm" ? available & ~1 : available;
            pending = available - length;
            bytes += read;
            if (length > 0) yield return new SpeechAudioPart { Audio = Convert.ToBase64String(buffer, 0, length), AudioFormat = new() { Type = mime, Rate = rate } };
            if (pending != 0) buffer[0] = buffer[length];
        }
        ValidateAudio(format, bytes);
        yield return new SpeechFinishPart { ProviderMetadata = Metadata(plan, mime, format, rate, bytes, responseHeaders) };
    }

    private async Task<HttpResponseMessage> SendSpeechAsync(Plan plan, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        using var request = new HttpRequestMessage(HttpMethod.Post, plan.Model == "coda" ? "https://coda.api.rime.ai/" : "v1/rime-tts")
        { Content = JsonContent.Create(plan.Body, options: Json) };
        NativeHttp.ApplyHeaders(request, headers);
        request.Headers.Authorization = new("Bearer", RequiredKey(credentials, Id));
        request.Headers.Accept.ParseAdd(plan.Accept);
        return await NativeHttp.SendAsync(client, request, ct);
    }

    private sealed record Plan(string Model, Dictionary<string, object?> Body, string Accept, string Format, int Rate, List<object> Warnings);

    private static Plan PlanFor(AudioSpeechRequest request, Dictionary<string, JsonElement> options, string? language = null, int? outputRate = null)
    {
        if (string.IsNullOrWhiteSpace(request.Input) || request.Input.EnumerateRunes().Count() > 1000)
            throw new ProviderRequestException("Rime HTTP speech requires 1–1,000 characters.");
        var parts = request.Model.Split('/', StringSplitOptions.TrimEntries);
        var model = parts[0].ToLowerInvariant();
        if (model is not ("coda" or "mistv3") || parts.Length > 2 || parts.Any(string.IsNullOrEmpty))
            throw new ProviderRequestException("Use Rime model coda or mistv3, optionally followed by /voice.", 422);
        var warnings = new List<object>();
        void Ignore(string feature, string reason) => warnings.Add(new { type = "ignored", feature, reason });
        var voice = (parts.Length == 2 ? parts[1] : null) ?? NonEmpty(request.Voice) ?? NonEmpty(Get<string>("voice"));
        if (parts.Length == 2 && ((NonEmpty(request.Voice) is { } requested && requested != voice) || (NonEmpty(Get<string>("voice")) is { } supplied && supplied != voice)))
            Ignore("voice", "Voice is derived from the model ID.");
        if (request.Speed is not null || Present("speedAlpha")) Ignore("speed", "Use timeScaleFactor.");
        if (!string.IsNullOrWhiteSpace(request.Instructions)) Ignore("instructions", "Rime HTTP speech does not support instructions.");
        foreach (var legacy in new[] { "noTextNormalization", "saveOovs" })
            if (Present(legacy)) Ignore(legacy, "Legacy Mist option.");
        var lang = (NonEmpty(language) ?? NonEmpty(Get<string>("language")) ?? "en").ToLowerInvariant() switch
        { "eng" => "en", "spa" => "es", "fra" or "fre" => "fr", "por" => "pt", "ger" or "deu" => "de",
            "jpn" => "ja", "ara" => "ar", "hin" => "hi", "ita" => "it", "nld" or "dut" => "nl", var normalized => normalized };
        var nativeRate = Get<int?>("samplingRate");
        var rate = outputRate ?? nativeRate ?? 24000;
        if (rate <= 0) throw new ProviderRequestException("Rime samplingRate must be positive.");
        if (outputRate is not null && nativeRate is not null && outputRate != nativeRate) Ignore("samplingRate", "outputFormat.rate takes precedence.");
        var (format, accept) = NormalizeFormat(request.ResponseFormat);
        var body = new Dictionary<string, object?> { ["modelId"] = model, ["text"] = request.Input, ["lang"] = lang, ["samplingRate"] = rate };
        if (voice is not null) body["speaker"] = voice;
        if (Get<double?>("timeScaleFactor") is { } factor)
        {
            if (!double.IsFinite(factor)) throw new ProviderRequestException("Rime timeScaleFactor must be finite.");
            body["timeScaleFactor"] = factor;
        }
        foreach (var (field, allowed) in new[] { ("pauseBetweenBrackets", model == "mistv3"), ("inlineSpeedAlpha", model == "mistv3"),
            ("phonemizeBetweenBrackets", model == "mistv3" && lang == "en"), ("customLexicon", model == "coda") })
        {
            if (!Present(field)) continue;
            if (!allowed) { Ignore(field, "Not supported for this model or language."); continue; }
            if (field == "customLexicon") ValidateLexicon(options[field], lang);
            else if (field == "inlineSpeedAlpha") { if (NonEmpty(Get<string>(field)) is null) continue; }
            else _ = Get<bool?>(field);
            body[field] = options[field];
        }
        foreach (var (key, value) in options)
            if (!Reserved.Contains(key)) body[key] = value;
        return new(model, body, accept, format, rate, warnings);

        bool Present(string name) => options.TryGetValue(name, out var value) && value.ValueKind != JsonValueKind.Null;
        T? Get<T>(string name)
        {
            try { return Present(name) ? options[name].Deserialize<T>(Json) : default; }
            catch (JsonException) { throw new ProviderRequestException($"Invalid Rime option '{name}'."); }
        }
    }

    private static void ValidateLexicon(JsonElement value, string lang)
    {
        if (value.ValueKind != JsonValueKind.Array || value.GetArrayLength() > 500 ||
            (value.GetArrayLength() > 0 && lang is not ("en" or "de" or "es" or "fr" or "it" or "pt")) ||
            value.EnumerateArray().Any(entry => NonEmpty(String(entry, "spelling")) is null || NonEmpty(String(entry, "pronunciation")) is null))
            throw new ProviderRequestException("Rime customLexicon supports up to 500 spelling/pronunciation pairs in en, de, es, fr, it, or pt.");
    }

    private static Dictionary<string, JsonElement> Options(Dictionary<string, JsonElement>? extras, bool flat = false)
    {
        var options = new Dictionary<string, JsonElement>(StringComparer.OrdinalIgnoreCase);
        if (extras?.TryGetValue(Id, out var native) == true)
        {
            if (native.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("Rime options must be an object.");
            foreach (var property in native.EnumerateObject()) options[property.Name] = property.Value;
        }
        if (flat && extras is not null)
            foreach (var (key, value) in extras)
                if (key != Id) options[key] = value;
        return options;
    }

    private static (string Mime, string Format, int Rate) ResponseFormat(HttpResponseMessage response, Plan plan)
    {
        var type = response.Content.Headers.ContentType;
        if (type is null || type.MediaType == "application/octet-stream") return (plan.Accept, plan.Format, plan.Rate);
        if (!type.MediaType!.StartsWith("audio/", StringComparison.OrdinalIgnoreCase)) throw new IOException("Rime did not return audio.");
        var (format, _) = NormalizeFormat(type.MediaType);
        var rate = type.Parameters.FirstOrDefault(p => p.Name.Equals("rate", StringComparison.OrdinalIgnoreCase))?.Value?.Trim('"');
        return (type.ToString(), format, int.TryParse(rate, out var actual) && actual > 0 ? actual : plan.Rate);
    }

    private static (string Format, string Mime) NormalizeFormat(string? format)
    {
        var value = NonEmpty(format)?.ToLowerInvariant();
        if (value?.Contains(';') == true)
        {
            if (!MediaTypeHeaderValue.TryParse(value, out var type) || type.Parameters.Any(p => p.Name != "codecs" || p.Value?.Trim('"') != "opus"))
                throw new ProviderRequestException("Rime only supports the opus codec parameter; set rates via samplingRate.", 422);
            value = type.MediaType;
        }
        return value switch
        {
            null or "mp3" or "audio/mpeg" or "audio/mp3" => ("mp3", "audio/mpeg"),
            "webm" or "opus" or "webm-opus" or "audio/webm" => ("webm", "audio/webm;codecs=opus"),
            "ogg" or "audio/ogg" => ("ogg", "audio/ogg;codecs=opus"),
            "wav" or "wave" or "audio/wav" or "audio/x-wav" => ("wav", "audio/wav"),
            "pcm" or "l16" or "audio/l16" or "audio/pcm" => ("pcm", "audio/L16"),
            "mulaw" or "mu-law" or "ulaw" or "pcmu" or "audio/pcmu" or "audio/x-mulaw" or "audio/basic" => ("mulaw", "audio/PCMU"),
            _ => throw new ProviderRequestException($"Unsupported Rime audio format '{format}'.", 422)
        };
    }

    private static Dictionary<string, string> ResponseHeaders(HttpResponseMessage response, string mime)
    {
        var headers = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase) { ["Content-Type"] = mime };
        foreach (var header in response.Headers)
            if (header.Key.StartsWith("x-rime-", StringComparison.OrdinalIgnoreCase) || header.Key.Equals("x-request-id", StringComparison.OrdinalIgnoreCase))
                headers[header.Key] = string.Join(", ", header.Value);
        return headers;
    }

    private static Dictionary<string, JsonElement> Metadata(Plan plan, string mime, string format, int rate, long bytes, Dictionary<string, string> headers) => new()
    {
        [Id] = JsonSerializer.SerializeToElement(new { model = plan.Model, voice = plan.Body.GetValueOrDefault("speaker"), language = plan.Body["lang"],
            audioFormat = format, accept = plan.Accept, samplingRate = rate, mimeType = mime, bytes, headers,
            timeScaleFactor = plan.Body.GetValueOrDefault("timeScaleFactor"), pauseBetweenBrackets = plan.Body.GetValueOrDefault("pauseBetweenBrackets") ?? false,
            phonemizeBetweenBrackets = plan.Body.GetValueOrDefault("phonemizeBetweenBrackets"), inlineSpeedAlpha = plan.Body.GetValueOrDefault("inlineSpeedAlpha") }, Json)
    };

    private static string? NonEmpty(string? value) => string.IsNullOrWhiteSpace(value) ? null : value.Trim();
    private static void ValidateAudio(string format, long bytes)
    {
        if (bytes == 0 || (format == "pcm" && bytes % 2 != 0)) throw new IOException("Rime returned empty or incomplete audio.");
    }
}
