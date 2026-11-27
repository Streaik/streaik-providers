using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class AskARCProviderRegistration
{
    public static IServiceCollection AddAskARC(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, "askarc", "https://api.askarc.app/api/v1/");
        return services.AddProvider<AskARCProvider>("askarc");
    }
}

internal sealed class AskARCProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, "askarc"), INativeChatOptionsProvider
{
    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (ConfiguredApiKey() is not { } key || string.IsNullOrWhiteSpace(key)) yield break;
        var result = await FetchModelsAsync(key, ct);
        foreach (var item in Array(Property(result, "data")))
        {
            if (String(item, "id") is not { Length: > 0 } id) continue;
            var context = Property(item, "contextWindow");
            yield return new Model
            {
                Id = id, Name = String(item, "name") ?? id, OwnedBy = String(item, "provider") ?? ProviderId,
                Type = "language", ContextWindow = context.ValueKind == JsonValueKind.Number && context.TryGetInt32(out var size) ? size : null
            };
        }
    }
}
