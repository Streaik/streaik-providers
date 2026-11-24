using System.Runtime.CompilerServices;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;

namespace Streaik.Providers;

public static class BananaPeelProviderRegistration
{
    public static IServiceCollection AddBananaPeel(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, BananaPeelProvider.Id, "https://bananapeel.com/api/v1/");
        return services.AddProvider<BananaPeelProvider>(BananaPeelProvider.Id, "anthropic-version", "anthropic-beta");
    }
}

internal sealed class BananaPeelProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : BearerTextEndpoints(clients, credentials, Id), IModelListingProvider
{
    internal const string Id = "bananapeel";

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        await Task.CompletedTask;
        if (!string.IsNullOrWhiteSpace(ConfiguredApiKey()))
            yield return new() { Id = "banana-peel", Name = "Banana Peel", Type = "language", OwnedBy = "BananaPeel" };
    }
}
