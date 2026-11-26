using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.OpenAI.Models;

namespace Streaik.Providers;

public static class ApekeyProviderRegistration
{
    public static IServiceCollection AddApekey(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, "apekey", "https://apekey.ai/v1/");
        return services.AddProvider<ApekeyProvider>("apekey");
    }
}

internal sealed class ApekeyProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, "apekey"), INativeChatOptionsProvider, IChatBridgeProvider
{
    public IChatCompletionsProvider ChatBridge { get; } = new FlatTextBridge(clients, credentials);

    public override IAsyncEnumerable<Model> ListModelsAsync(CancellationToken ct)
        => StaticModelListing.ListAsync(ProviderId, ConfiguredApiKey(), ct);

    private sealed class FlatTextBridge(IHttpClientFactory clients, IProviderCredentials credentials)
        : CompatibleChatProvider(clients, credentials, "apekey"), INativeChatOptionsProvider
    {
        protected override void PrepareCompatibleRequest(ChatCompletionOptions request)
        {
            base.PrepareCompatibleRequest(request);
            foreach (var message in request.Messages)
            {
                if (message.Content.Parts is not { } parts) continue;
                if (parts.Any(part => part is not ChatMessageTextPart))
                    throw new ProviderRequestException("Apekey bridged requests support only text content.", 422);
                message.Content = string.Join("\n", parts.OfType<ChatMessageTextPart>().Select(part => part.Text));
            }
        }
    }
}
