using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.Messages;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class CognitivessAIProviderRegistration
{
    public static IServiceCollection AddCognitivessAI(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, CognitivessAIProvider.Id, "https://api.cognitivess.com/v1/");
        return services.AddProvider<CognitivessAIProvider>(CognitivessAIProvider.Id, "anthropic-version", "anthropic-beta");
    }
}

internal sealed class CognitivessAIProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), IResponsesProvider, IMessagesProvider, INativeChatOptionsProvider
{
    internal const string Id = "cognitivessai";
    private readonly BearerTextEndpoints text = new(clients, credentials, Id);

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        // Model discovery is public, even when inference is authenticated.
        using var client = CreateClient();
        var body = await NativeHttp.GetAsync<JsonElement>(client, "models", _ => { }, ct);
        foreach (var item in Array(Property(body, "data")))
        {
            if (String(item, "id") is not { Length: > 0 } model) continue;
            var pricing = Property(item, "pricing");
            var rates = ProviderPrices.ReadPair(pricing, "prompt", "completion");
            if (rates is { Input: > 0, Output: > 0 })
            {
                rates.Input /= 1_000_000m;
                rates.Output /= 1_000_000m;
                if (ProviderPrices.TokenPrice(Property(pricing, "cached_prompt")) is > 0 and var cacheRead)
                    rates.InputCacheRead = cacheRead / 1_000_000m;
            }
            else rates = null;
            yield return new()
            {
                Id = model, Name = String(item, "name") ?? model, Type = String(item, "type") ?? "model",
                OwnedBy = String(item, "owned_by") ?? Id, Description = String(item, "description"),
                ContextWindow = Limit(item, "context_length"), MaxTokens = Limit(item, "max_output"), Pricing = rates
            };
        }
    }

    private static int? Limit(JsonElement item, string key)
        => Property(item, key) is { ValueKind: JsonValueKind.Number } number && number.TryGetInt32(out var limit) ? limit : null;

    public Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.CreateResponseAsync(request, headers, ct);

    public IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.StreamResponseAsync(request, headers, ct);

    public Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.CreateMessageAsync(request, headers, ct);

    public IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.StreamMessageAsync(request, headers, ct);
}
