using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class TensorBlockProviderRegistration
{
    public static IServiceCollection AddTensorBlock(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, TensorBlockProvider.Id, "https://api.forge.tensorblock.co/v1/");
        ProviderServiceCollectionExtensions.AddClient(services, TensorBlockProvider.AssetClient, "https://api.forge.tensorblock.co/");
        return services.AddProvider<TensorBlockProvider>(TensorBlockProvider.Id, "anthropic-version", "anthropic-beta");
    }
}

internal sealed partial class TensorBlockProvider(IHttpClientFactory clients, IProviderCredentials credentials, IConfiguration? configuration = null)
    : BearerTextEndpoints(clients, credentials, Id), IChatCompletionsProvider, IModelListingProvider, IImageProvider
{
    internal const string Id = "tensorblock";
    internal const string AssetClient = "tensorblock-assets";

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (ConfiguredApiKey() is not { } key || string.IsNullOrWhiteSpace(key)) yield break;
        using var client = CreateClient();
        var body = await NativeHttp.GetAsync<JsonElement>(client, "models", h => h.Authorization = new("Bearer", key), ct);
        foreach (var item in Array(Property(body, "data")))
        {
            if (String(item, "id") is not { Length: > 0 } model) continue;
            var context = Property(item, "context_length");
            yield return new Model { Id = model, Name = String(item, "name") ?? model,
                Description = String(item, "description"), OwnedBy = String(item, "owned_by") ?? Id, Type = String(item, "type") ?? "model",
                ContextWindow = context.ValueKind == JsonValueKind.Number && context.TryGetInt32(out var count) ? count : null };
        }
    }

    public async Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        return await NativeHttp.PostAsync<Dictionary<string, JsonElement>, ChatCompletion>(client, "chat/completions",
            ChatBody(request, false), Auth(), headers, Json, ct);
    }

    public async IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        await foreach (var part in NativeHttp.StreamAsync<Dictionary<string, JsonElement>, ChatCompletionUpdate>(client, "chat/completions",
            ChatBody(request, true), Auth(), headers, Json, part => part.AdditionalProperties?.ContainsKey("error") == true, ct)) yield return part;
    }

    private static Dictionary<string, JsonElement> ChatBody(ChatCompletionOptions request, bool stream)
    {
        var body = NativeOptionsPayload.Build(request, Id, Json, stream);
        if (request.Store == false) body.Remove("store");
        if (stream && request.StreamOptions is null)
            body["stream_options"] = JsonSerializer.SerializeToElement(new { include_usage = true });
        return body;
    }
}
