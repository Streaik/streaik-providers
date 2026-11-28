using System.Globalization;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class SpeechactorsProviderRegistration
{
    public static IServiceCollection AddSpeechactors(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, SpeechactorsProvider.Id, "https://api.speechactors.com/v1/");
        return services.AddProvider<SpeechactorsProvider>(SpeechactorsProvider.Id);
    }
}

internal sealed class SpeechactorsProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IStreamingSpeechProvider, IChatAudioRoutingProvider
{
    internal const string Id = "speechactors";
    private static readonly HashSet<string> Reserved = new(StringComparer.OrdinalIgnoreCase)
        { "model", "input", "text", "voice", "response_format", "stream_format", "instructions", "speed" };

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var key = credentials.GetApiKey(Id);
        if (string.IsNullOrWhiteSpace(key)) yield break;
        using var client = clients.CreateClient(Id);
        var auth = NativeHttp.Bearer(key);
        var voices = await NativeHttp.GetAsync<JsonElement>(client, "voices", auth, ct);
        var locales = await NativeHttp.GetAsync<JsonElement>(client, "languages", auth, ct);
        var languages = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var item in Entries(locales, "languages"))
            if (String(item, "code")?.Trim() is { Length: > 0 } code && String(item, "language")?.Trim() is { Length: > 0 } name)
                languages[code] = name;

        yield return new Model { Id = "tts", OwnedBy = "Speechactors", Type = "speech", Name = "Speechactors TTS" };
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var item in Entries(voices, "voices"))
        {
            if (String(item, "vid")?.Trim() is not { Length: > 0 } voice || voice.Contains('/')) continue;
            var locale = String(item, "locale")?.Trim();
            if (string.IsNullOrEmpty(locale) && voice.LastIndexOf('-') is > 0 and var separator) locale = voice[..separator];
            if (string.IsNullOrEmpty(locale) || locale.Contains('/') || !seen.Add($"{voice}/{locale}")) continue;
            var language = languages.GetValueOrDefault(locale) ?? String(item, "locale_name") ?? String(item, "localeName") ?? locale;
            var name = String(item, "name") ?? voice;
            var gender = String(item, "gender")?.Trim();
            yield return new Model
            {
                Id = $"tts/{voice}/{locale}", OwnedBy = "Speechactors", Type = "speech",
                Name = $"{name}, {(string.IsNullOrEmpty(gender) ? "Unknown" : gender)}, {language}",
                Description = "Speechactors TTS.", Tags = string.IsNullOrEmpty(gender) ? ["voice", locale] : ["voice", locale, gender]
            };
        }
    }

    public bool IsSpeechModel(string model) => SplitModel(model) is not null;
    public bool IsTranscriptionModel(string model) => false;

    // The upstream returns MP3 bytes; the gateway supplies SSE framing when requested.
    public Task<HttpResponseMessage> SpeakStreamingAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeakAsync(request, headers, ct);

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var model = SplitModel(request.Model) ?? throw new ProviderRequestException("Use tts, tts/{voiceId}, or tts/{voiceId}/{locale}.");
        if (string.IsNullOrWhiteSpace(request.Input)) throw new ProviderRequestException("Speechactors requires nonempty text.");
        var fields = NativeMediaOptions.Read(Id, request.AdditionalProperties, Reserved);
        var voice = (model.ElementAtOrDefault(1) ?? request.Voice ?? Text(fields, "vid"))?.Trim();
        var locale = (model.ElementAtOrDefault(2) ?? Text(fields, "language") ?? Text(fields, "locale"))?.Trim();
        if (string.IsNullOrEmpty(voice) || string.IsNullOrEmpty(locale)) throw new ProviderRequestException("Speechactors requires a voice ID and locale.");
        var warnings = new List<object>();
        if (!string.IsNullOrWhiteSpace(request.Instructions)) warnings.Add(new { type = "unsupported", feature = "instructions" });
        if (request.Speed.HasValue) warnings.Add(new { type = "ignored", feature = "speed", reason = "use speakingRate" });
        SetControl("speakingRate", -100, 200);
        SetControl("pitch", -50, 50);
        fields.Remove("vid");
        fields.Remove("locale");
        fields.Remove("language");
        fields["vid"] = JsonSerializer.SerializeToElement(voice);
        fields["locale"] = JsonSerializer.SerializeToElement(locale);
        fields["text"] = JsonSerializer.SerializeToElement(request.Input);

        using var client = clients.CreateClient(Id);
        using var message = new HttpRequestMessage(HttpMethod.Post, "generate") { Content = JsonContent.Create(fields, options: Json) };
        message.Headers.Accept.ParseAdd("audio/*");
        message.Headers.Authorization = new("Bearer", RequiredKey(credentials, Id));
        NativeHttp.ApplyHeaders(message, headers);
        using var upstream = await NativeHttp.SendAsync(client, message, ct);
        var audio = upstream.Content;
        if (audio.Headers.ContentType?.MediaType is not (null or "application/octet-stream" or "audio/mpeg" or "audio/mp3"))
            throw new IOException("Speechactors did not return MP3 audio.");
        audio.Headers.ContentType = new("audio/mpeg");
        var raw = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, values) in upstream.Headers.Concat(audio.Headers)) raw[name] = string.Join(", ", values);
        var safe = NativeHttp.SafeResponseMetadataHeaders(raw);
        var result = new SpeechProviderResponse(audio, NativeMediaOptions.Metadata(Id, new { headers = safe, warnings }));
        upstream.Content = null; // The gateway owns the stream now.
        return result;

        void SetControl(string field, int min, int max)
        {
            if (!fields.Remove(field, out var value)) return;
            var number = 0;
            var valid = value.ValueKind == JsonValueKind.Number ? value.TryGetInt32(out number)
                : value.ValueKind == JsonValueKind.String && int.TryParse(value.GetString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out number);
            if (valid && number >= min && number <= max) fields[field] = JsonSerializer.SerializeToElement(number);
            else warnings.Add(new { type = "ignored", feature = field, reason = $"expected an integer in [{min},{max}]" });
        }
    }

    private static string[]? SplitModel(string model)
    {
        var parts = model.Split('/', StringSplitOptions.TrimEntries);
        return parts.Length is >= 1 and <= 3 && parts[0].Equals("tts", StringComparison.OrdinalIgnoreCase)
            && parts.All(part => part.Length > 0) ? parts : null;
    }

    private static IEnumerable<JsonElement> Entries(JsonElement body, string kind)
        => body.ValueKind == JsonValueKind.Array ? Array(body)
            : new[] { "data", "items", kind }.Select(name => Property(body, name))
                .Where(item => item.ValueKind == JsonValueKind.Array).Take(1).SelectMany(Array);

    private static string? Text(Dictionary<string, JsonElement> fields, string name)
        => fields.TryGetValue(name, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
}
