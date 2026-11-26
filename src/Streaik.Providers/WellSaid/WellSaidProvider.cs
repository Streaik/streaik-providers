using System.Globalization;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class WellSaidProviderRegistration
{
    public static IServiceCollection AddWellSaid(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, WellSaidProvider.Id, "https://api.wellsaidlabs.com/v1/tts/");
        return services.AddProvider<WellSaidProvider>(WellSaidProvider.Id);
    }
}

internal sealed class WellSaidProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IStreamingSpeechProvider, IChatAudioRoutingProvider
{
    internal const string Id = "wellsaid";
    private static readonly HashSet<string> Reserved = new(StringComparer.OrdinalIgnoreCase)
    {
        "text", "speaker_id", "model", "input", "voice", "response_format", "stream_format", "stream", "providerOptions",
        "providerMetadata", Id, "authorization", "api-key", "api_key", "apiKey", "x-api-key", "headers"
    };

    public bool IsSpeechModel(string model) => TrySpeakerId(model, out _);
    public bool IsTranscriptionModel(string model) => false;

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var key = credentials.GetApiKey(Id);
        if (string.IsNullOrWhiteSpace(key)) yield break;
        using var client = clients.CreateClient(Id);
        var body = await NativeHttp.GetAsync<JsonElement>(client, "avatars", h => h.Add("X-Api-Key", key), ct);
        if (Property(body, "avatars") is not { ValueKind: JsonValueKind.Array } avatars)
            throw new JsonException("Expected a WellSaid avatars array.");
        foreach (var item in avatars.EnumerateArray())
        {
            if (Property(item, "id") is not { ValueKind: JsonValueKind.Number } id || !id.TryGetInt32(out var speakerId) || speakerId < 0)
                continue;
            var model = speakerId.ToString(CultureInfo.InvariantCulture);
            yield return new() { Id = model, Name = String(item, "name") ?? model, OwnedBy = Id, Type = "speech",
                Description = String(item, "description") };
        }
    }

    public Task<HttpResponseMessage> SpeakStreamingAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeakAsync(request, headers, ct);

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!TrySpeakerId(request.Model, out var speakerId))
            throw new ProviderRequestException("WellSaid models must use a native integer avatar ID.");
        if (string.IsNullOrWhiteSpace(request.Input)) throw new ProviderRequestException("WellSaid requires text input.");
        if (request.Speed is not null and not 1d || request.Instructions is not null)
            throw new ProviderRequestException("WellSaid does not support standard speed or instructions; use native provider options.");
        if (request.ResponseFormat is { } format && !format.Equals("mp3", StringComparison.OrdinalIgnoreCase))
            throw new ProviderRequestException("WellSaid only supports MP3 audio.");

        var payload = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        var options = request.AdditionalProperties;
        foreach (var (name, value) in options ?? [])
            if (!Reserved.Contains(name)) payload[name] = value;
        if (options?.TryGetValue(Id, out var scoped) == true) AddOptions(scoped);
        if (options?.TryGetValue("providerOptions", out var envelope) == true)
        {
            if (envelope.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("providerOptions must be an object.");
            if (envelope.TryGetProperty(Id, out var native)) AddOptions(native);
        }
        payload["text"] = JsonSerializer.SerializeToElement(request.Input);
        payload["speaker_id"] = JsonSerializer.SerializeToElement(speakerId);

        using var client = clients.CreateClient(Id);
        using var message = new HttpRequestMessage(HttpMethod.Post, "stream") { Content = JsonContent.Create(payload) };
        message.Headers.Add("X-Api-Key", RequiredKey(credentials, Id));
        message.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("audio/mpeg"));
        using var response = await NativeHttp.SendAsync(client, message, ct);
        if (!string.Equals(response.Content.Headers.ContentType?.MediaType, "audio/mpeg", StringComparison.OrdinalIgnoreCase)
            || response.Content.Headers.ContentLength == 0)
            throw new JsonException("Expected a non-empty WellSaid MP3 response.");
        var responseHeaders = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, values) in response.Headers.Concat(response.Content.Headers)) responseHeaders[name] = string.Join(", ", values);
        var safe = NativeHttp.SafeResponseMetadataHeaders(responseHeaders);
        var result = new SpeechProviderResponse(response.Content,
            new() { [Id] = JsonSerializer.SerializeToElement(new { headers = safe }, Json) });
        response.Content = null; // The gateway owns the audio stream.
        return result;

        void AddOptions(JsonElement native)
        {
            if (native.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("WellSaid options must be an object.");
            foreach (var property in native.EnumerateObject())
                if (!Reserved.Contains(property.Name)) payload[property.Name] = property.Value;
        }
    }

    private static bool TrySpeakerId(string model, out int speakerId)
        => int.TryParse(model, NumberStyles.None, CultureInfo.InvariantCulture, out speakerId);
}
