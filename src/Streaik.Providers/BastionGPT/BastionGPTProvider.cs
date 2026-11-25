using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.OpenAI.Models;

namespace Streaik.Providers;

public static class BastionGPTProviderRegistration
{
    public static IServiceCollection AddBastionGPT(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, BastionGPTProvider.Id, "https://api.bastiongpt.com/");
        return services.AddProvider<BastionGPTProvider>(BastionGPTProvider.Id);
    }
}

internal sealed class BastionGPTProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IChatCompletionsProvider, IModelListingProvider, INativeChatOptionsProvider
{
    internal const string Id = "bastiongpt";
    private const string ChatPath = "v1/ChatCompletion";

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        ct.ThrowIfCancellationRequested();
        if (!string.IsNullOrWhiteSpace(credentials.GetApiKey(Id)))
            yield return new Model { Id = Id, Name = "BastionGPT", OwnedBy = "BastionGPT", Type = "language",
                ContextWindow = 500_000, Description = "Purpose-built for therapy, clinical, and healthcare workflows." };
    }

    public async Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        return await NativeHttp.PostAsync<Dictionary<string, JsonElement>, ChatCompletion>(client, ChatPath,
            Body(request, false), Auth(), headers, MediaHttp.Json, ct);
    }

    public async IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        await foreach (var part in NativeHttp.StreamAsync<Dictionary<string, JsonElement>, ChatCompletionUpdate>(client, ChatPath,
            Body(request, true), Auth(), headers, MediaHttp.Json,
            part => part.AdditionalProperties?.ContainsKey("error") == true, ct)) yield return part;
    }

    private static Dictionary<string, JsonElement> Body(ChatCompletionOptions request, bool stream)
    {
        var body = NativeOptionsPayload.Build(request, Id, MediaHttp.Json, stream);
        body.Remove("model"); // The upstream selects the model for the API key.
        if (request.Store == false) body.Remove("store");
        if (stream && request.StreamOptions is null)
            body["stream_options"] = JsonSerializer.SerializeToElement(new { include_usage = true });
        return body;
    }

    private Action<HttpRequestHeaders> Auth()
    {
        var key = MediaHttp.RequiredKey(credentials, Id);
        return headers => headers.Add("Key", key);
    }
}
