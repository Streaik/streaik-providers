using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;

namespace Streaik.Providers;

public static class AkumiProviderRegistration
{
    public static IServiceCollection AddAkumi(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AkumiProvider.Id, "https://api.akumi.cloud/");
        return services.AddProvider<AkumiProvider>(AkumiProvider.Id);
    }
}

internal sealed class AkumiProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatEmbeddingProvider(clients, credentials, Id, "v1/")
{
    internal const string Id = "akumi";

    protected override void PrepareCompatibleRequest(ChatCompletionOptions request)
    {
        base.PrepareCompatibleRequest(request);
        foreach (var message in request.Messages)
        {
            // Flatten text-only history while retaining native multimodal content on direct OpenAI requests.
            if (message.Content.Kind == ChatMessageContentKind.Parts && message.Content.Parts!.All(part => part is ChatMessageTextPart))
                message.Content = ChatMessageContentExtensions.ToText(message.Content)!;
        }
    }
}
