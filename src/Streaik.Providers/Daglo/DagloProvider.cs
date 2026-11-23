using System.Net.Http.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Daglo;

public static class DagloRegistration
{
    public static IServiceCollection AddDaglo(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, DagloProvider.Id, "https://apis.daglo.ai/");
        return services.AddProvider<DagloProvider>(DagloProvider.Id);
    }
}

internal sealed partial class DagloProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id, "nlp/v1/sync/"), ISpeechProvider, ITranscriptionProvider, INativeAudioTranscriptionProvider
{
    internal const string Id = "daglo";

    public override IAsyncEnumerable<Model> ListModelsAsync(CancellationToken ct)
        => (string.IsNullOrWhiteSpace(ConfiguredApiKey()) ? [] : ListModels()).ToAsyncEnumerable();

    private static IEnumerable<Model> ListModels()
    {
        yield return new() { Id = "elli", Name = "elli", Type = "language", OwnedBy = Id };
        yield return new() { Id = "tts", Name = "tts", Type = "speech", OwnedBy = Id };
        yield return new() { Id = "stt", Name = "stt", Type = "transcription", OwnedBy = Id };
    }

    public override Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => base.CompleteChatAsync(FlattenContent(request), headers, ct);

    public override IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => base.StreamChatAsync(FlattenContent(request), headers, ct);

    private static ChatCompletionOptions FlattenContent(ChatCompletionOptions request)
    {
        foreach (var message in request.Messages)
        {
            if (message.Content.Kind != ChatMessageContentKind.Parts) continue;
            if (message.Content.Parts!.Any(part => part is not ChatMessageTextPart))
                throw new ProviderRequestException("Daglo chat accepts only text content.", 422);
            message.Content = ChatMessageContentExtensions.ToText(message.Content)!;
        }
        return request;
    }

    public async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        var upstream = await SendAsync(client, HttpMethod.Post, "tts/v1/sync/audios", JsonContent.Create(new { text = request.Input }, options: Json),
            "Bearer", ApiKey(), headers, ct);
        upstream.Content.Headers.ContentType ??= new("audio/wav");
        return upstream;
    }
}
