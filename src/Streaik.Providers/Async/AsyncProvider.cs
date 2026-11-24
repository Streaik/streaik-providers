using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class AsyncProviderRegistration
{
    public static IServiceCollection AddAsyncAI(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AsyncProvider.Id, "https://api.async.com/");
        return services.AddProvider<AsyncProvider>(AsyncProvider.Id);
    }
}

internal sealed partial class AsyncProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, ISpeechProvider, IStreamingSpeechEventsProvider, IChatAudioRoutingProvider,
      ITranscriptionProvider, INativeAudioTranscriptionProvider
{
    internal const string Id = "async";
    private const string Asr = "async_asr_v1.0";
    private static readonly (string Id, string Name)[] SpeechModels =
    [
        ("async_pro_v1.0", "Async Pro v1.0"),
        ("async_flash_v1.5", "Async Flash v1.5"),
        ("async_flash_v1.0", "Async Flash v1.0")
    ];

    public bool IsSpeechModel(string model) => SpeechModel(model).Id is not null;
    public bool IsTranscriptionModel(string model) => string.Equals(model, Asr, StringComparison.OrdinalIgnoreCase);

    private static (string? Id, string? Voice) SpeechModel(string model)
    {
        var parts = model.Split('/', 2);
        foreach (var (id, _) in SpeechModels)
            if (parts[0].Equals(id, StringComparison.OrdinalIgnoreCase))
                return (id, parts.Length == 2 ? parts[1].Trim() : null);
        return default;
    }

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        yield return new Model { Id = Asr, Name = "Async ASR v1.0", OwnedBy = Id, Type = "transcription" };
        foreach (var (id, name) in SpeechModels)
            yield return new Model { Id = id, Name = name, OwnedBy = Id, Type = "speech" };

        using var client = clients.CreateClient(Id);
        foreach (var (id, name) in SpeechModels)
        {
            var cursors = new HashSet<string>(StringComparer.Ordinal);
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            string? cursor = null;
            for (var i = 0; i < 10; i++)
            {
                var body = new JsonObject { ["model_id"] = id, ["limit"] = 100 };
                if (cursor is not null) body["after"] = cursor;
                JsonElement page;
                try
                {
                    using var response = await SendAsync(client, "voices", JsonContent.Create(body, options: Json),
                        new Dictionary<string, string>(), ct);
                    page = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
                }
                catch (HttpRequestException) when (!ct.IsCancellationRequested) { break; }
                foreach (var voice in Array(Property(page, "voices")))
                {
                    var voiceId = (String(voice, "voice_id") ?? String(voice, "voiceId") ?? String(voice, "id"))?.Trim();
                    if (string.IsNullOrEmpty(voiceId) || voiceId.Contains('/') || !seen.Add(voiceId)) continue;
                    yield return new Model { Id = $"{id}/{voiceId}", Name = $"{name} / {String(voice, "name") ?? voiceId}",
                        OwnedBy = Id, Type = "speech", Description = String(voice, "description") };
                }
                cursor = String(page, "next_cursor");
                if (string.IsNullOrWhiteSpace(cursor) || !cursors.Add(cursor)) break;
            }
        }
    }

    private async Task<HttpResponseMessage> SendAsync(HttpClient client, string path, HttpContent content,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct, string? accept = null)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, path) { Content = content };
        NativeHttp.ApplyHeaders(request, headers);
        request.Headers.Add("x-api-key", RequiredKey(credentials, Id));
        request.Headers.Add("version", "v1");
        if (accept is not null) request.Headers.Accept.ParseAdd(accept);
        return await NativeHttp.SendAsync(client, request, ct);
    }

    // Transport and routing fields take precedence; unknown provider fields retain their JSON types.
    private static JsonObject Options(IReadOnlyDictionary<string, JsonElement>? options, HashSet<string> reserved)
    {
        var result = new JsonObject();
        foreach (var (name, value) in options ?? new Dictionary<string, JsonElement>())
            if (!reserved.Contains(name)) result[name] = Copy(value);
        return result;
    }
}
