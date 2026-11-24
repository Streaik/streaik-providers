using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.Messages;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class TetrateProviderRegistration
{
    public static IServiceCollection AddTetrate(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, TetrateProvider.Id, "https://api.router.tetrate.ai/v1/");
        return services.AddProvider<TetrateProvider>(TetrateProvider.Id, "anthropic-version", "anthropic-beta");
    }
}

internal sealed class TetrateProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), IResponsesProvider, IMessagesProvider, INativeChatOptionsProvider
{
    internal const string Id = "tetrate";

    public async Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        return await NativeHttp.PostAsync<ResponseRequest, ResponseResult>(client, "responses", request, Bearer(ApiKey()), headers, ResponseJson.Default, ct);
    }

    public async IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        await foreach (var part in NativeHttp.StreamAsync<ResponseRequest, ResponseStreamPart>(client, "responses", request, Bearer(ApiKey()), headers,
            ResponseJson.Default, part => part.Type is "response.completed" or "response.failed" or "response.incomplete" or "error", ct))
            yield return part;
    }

    public async Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        return await NativeHttp.PostAsync<MessagesRequest, MessagesResponse>(client, "messages", request, Bearer(ApiKey()), headers, MessagesJson.Default, ct);
    }

    public async IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        await foreach (var part in NativeHttp.StreamAsync<MessagesRequest, MessageStreamPart>(client, "messages", request, Bearer(ApiKey()), headers,
            MessagesJson.Default, part => part.Type is "message_stop" or "error", ct)) yield return part;
    }

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (ConfiguredApiKey() is not { } key || string.IsNullOrWhiteSpace(key)) yield break;
        using var client = CreateClient();
        var page = await NativeHttp.GetAsync<JsonElement>(client, "models", Bearer(key), ct);
        foreach (var item in Array(page.ValueKind == JsonValueKind.Array ? page : Property(page, "data")))
        {
            if (String(item, "id") is not { Length: > 0 } model) continue;
            var created = Property(item, "created");
            yield return new Model
            {
                Id = model, Name = String(item, "name") ?? model, OwnedBy = String(item, "owned_by") ?? Id,
                Description = String(item, "description"), Type = String(item, "type") ?? "language",
                Created = created.ValueKind == JsonValueKind.Number && created.TryGetInt64(out var epoch) ? epoch : null,
                ContextWindow = Integer(item, "context_window"), MaxTokens = Integer(item, "max_output_tokens"), Pricing = Pricing(item)
            };
        }
    }

    protected override ModelPricing? Pricing(JsonElement model)
        => Price(model, "input_price") is > 0 and var input && Price(model, "output_price") is > 0 and var output
            ? new() { Input = input, Output = output, InputCacheRead = Price(model, "cached_price"), InputCacheWrite = Price(model, "caching_price") }
            : null;

    private static int? Integer(JsonElement entry, string key)
        => Property(entry, key) is { ValueKind: JsonValueKind.Number } value && value.TryGetInt32(out var number) ? number : null;

    private static decimal? Price(JsonElement entry, string key) => Property(entry, key) switch
    {
        { ValueKind: JsonValueKind.Number } value when value.TryGetDecimal(out var number) => number,
        { ValueKind: JsonValueKind.String } value when decimal.TryParse(value.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var number) => number,
        _ => null
    };
}
