using System.Globalization;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class ResembleAIProviderRegistration
{
    public static IServiceCollection AddResembleAI(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, ResembleAIProvider.Id, "https://app.resemble.ai/");
        return services.AddProvider<ResembleAIProvider>(ResembleAIProvider.Id);
    }
}

internal sealed partial class ResembleAIProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, ISpeechProvider, IStreamingSpeechEventsProvider, ITranscriptionProvider, IChatAudioRoutingProvider
{
    internal const string Id = "resembleai";
    private const string TranscriptionModel = "speech-to-text";
    private static readonly (string Id, string Name)[] SpeechModels =
        [("chatterbox-turbo", "Chatterbox-Turbo"), ("chatterbox", "Chatterbox"), ("chatterbox-multilingual", "Chatterbox-Multilingual")];

    public bool IsSpeechModel(string model) => SpeechModels.Any(item => item.Id.Equals(model.Split('/', 2)[0], StringComparison.OrdinalIgnoreCase));
    public bool IsTranscriptionModel(string model) => model.Equals(TranscriptionModel, StringComparison.OrdinalIgnoreCase);

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (credentials.GetApiKey(Id) is not { Length: > 0 } key || string.IsNullOrWhiteSpace(key)) yield break;
        yield return new() { Id = TranscriptionModel, Name = TranscriptionModel, Type = "transcription", OwnedBy = "ResembleAI" };
        foreach (var (model, name) in SpeechModels)
            yield return new() { Id = model, Name = name, Type = "speech", OwnedBy = "ResembleAI" };

        using var client = clients.CreateClient(Id);
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        for (var page = 1; ; page++)
        {
            var path = "api/v2/voices?page_size=1000&page=" + page.ToString(CultureInfo.InvariantCulture);
            var body = await NativeHttp.GetAsync<JsonElement>(client, path, h => h.Authorization = new("Bearer", key), ct);
            foreach (var voice in Array(Property(body, "items")))
            {
                if (String(voice, "uuid")?.Trim() is not { Length: > 0 } uuid || uuid.Contains('/')
                    || !string.Equals(String(voice, "voice_status"), "ready", StringComparison.OrdinalIgnoreCase)
                    || Property(Property(voice, "api_support"), "sync").ValueKind != JsonValueKind.True || !seen.Add(uuid)) continue;
                var language = String(voice, "default_language");
                foreach (var (model, name) in SpeechModels)
                    yield return new() { Id = $"{model}/{uuid}", Name = $"{name} {String(voice, "name") ?? uuid} ({uuid})",
                        Description = $"ResembleAI {model} voice {uuid}.", Type = "speech", OwnedBy = "ResembleAI",
                        Tags = string.IsNullOrWhiteSpace(language) ? ["voice"] : ["voice", language.Replace('_', '-')] };
            }
            var pages = Property(body, "num_pages");
            if (!(pages.ValueKind == JsonValueKind.Number && pages.TryGetInt32(out var total)
                || pages.ValueKind == JsonValueKind.String && int.TryParse(pages.GetString(), NumberStyles.None, CultureInfo.InvariantCulture, out total))
                || page >= total || page == int.MaxValue) yield break;
        }
    }

    private Action<HttpRequestHeaders> Auth()
    {
        var key = RequiredKey(credentials, Id);
        return h => h.Authorization = new("Bearer", key);
    }

    private static Dictionary<string, string> SafeHeaders(HttpResponseMessage response)
        => NativeHttp.SafeResponseMetadataHeaders(response.Headers.Concat(response.Content.Headers)
            .GroupBy(h => h.Key, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(h => h.Key, h => (string?)string.Join(", ", h.SelectMany(x => x.Value)), StringComparer.OrdinalIgnoreCase))
            .Where(h => h.Value is not null).ToDictionary(h => h.Key, h => h.Value!, StringComparer.OrdinalIgnoreCase);
}
