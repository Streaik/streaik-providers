using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.Messages;
using static Streaik.Providers.MediaHttp;
using Model = Streaik.OpenAI.Models.Model;

namespace Streaik.Providers;

public static class WaferProviderRegistration
{
    public static IServiceCollection AddWafer(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, WaferProvider.Id, "https://pass.wafer.ai/v1/");
        return services.AddProvider<WaferProvider>(WaferProvider.Id, "anthropic-version", "anthropic-beta");
    }
}

internal sealed class WaferProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), IMessagesProvider, INativeChatOptionsProvider
{
    internal const string Id = "wafer";

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (ConfiguredApiKey() is not { Length: > 0 } key || string.IsNullOrWhiteSpace(key)) yield break;
        using var client = CreateClient();
        var body = await NativeHttp.GetAsync<JsonElement>(client, "models", Bearer(key), ct);
        foreach (var item in Array(Property(body, "data")))
        {
            if (String(item, "id") is not { Length: > 0 } id) continue;
            var wafer = Property(item, "wafer");
            yield return new Model
            {
                Id = id, OwnedBy = String(item, "owned_by") ?? Id, Type = "language",
                Object = String(item, "object") ?? "model", Created = ProviderModelJson.ReadCreated(item),
                Name = String(wafer, "display_name") ?? String(item, "name") ?? id,
                Description = String(wafer, "description") ?? String(item, "description"),
                ContextWindow = Property(wafer, "context_length") is { ValueKind: JsonValueKind.Number } context
                    && context.TryGetInt32(out var size) ? size : null
            };
        }
    }

    public async Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        return await NativeHttp.PostAsync<MessagesRequest, MessagesResponse>(client, "messages", request, Bearer(ApiKey()), headers, MessagesJson.Default, ct);
    }

    public async IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        await foreach (var part in NativeHttp.StreamAsync<MessagesRequest, MessageStreamPart>(client, "messages", request,
            Bearer(ApiKey()), headers, MessagesJson.Default, part => part.Type is "message_stop" or "error", ct)) yield return part;
    }
}
