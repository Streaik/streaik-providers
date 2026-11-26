using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class AIBramhaProviderRegistration
{
    public static IServiceCollection AddAIBramha(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AIBramhaProvider.Id, "https://www.aibramha.in/api/v1/");
        return services.AddProvider<AIBramhaProvider>(AIBramhaProvider.Id);
    }
}

internal sealed class AIBramhaProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), INativeChatOptionsProvider
{
    internal const string Id = "aibramha";

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (ConfiguredApiKey() is not { } key || string.IsNullOrWhiteSpace(key)) yield break;
        var response = await FetchModelsAsync(key, ct);
        foreach (var entry in Array(Property(response, "models")))
        {
            ct.ThrowIfCancellationRequested();
            if (String(entry, "id") is not { Length: > 0 } model) continue;
            var name = String(entry, "name") ?? model;
            var category = String(entry, "category");
            var context = Property(entry, "context_length");
            yield return new()
            {
                Id = model,
                Name = name,
                Description = String(entry, "description"),
                Type = !string.IsNullOrWhiteSpace(category) ? category : name.Contains("[IMAGE_GEN]", StringComparison.Ordinal) ? "image" : "language",
                OwnedBy = String(entry, "provider") ?? Id,
                ContextWindow = context.ValueKind == JsonValueKind.Number && context.TryGetInt32(out var size) ? size : null
            };
        }
    }
}
