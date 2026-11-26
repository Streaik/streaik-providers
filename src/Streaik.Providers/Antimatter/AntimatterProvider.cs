using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class AntimatterProviderRegistration
{
    public static IServiceCollection AddAntimatter(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AntimatterProvider.Id, "https://useantimatter.com/v1/");
        return services.AddProvider<AntimatterProvider>(AntimatterProvider.Id);
    }
}

internal sealed class AntimatterProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), INativeChatOptionsProvider
{
    internal const string Id = "antimatter";

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (ConfiguredApiKey() is not { Length: > 0 } key || string.IsNullOrWhiteSpace(key)) yield break;
        var catalog = Property(await FetchModelsAsync(key, ct), "data");
        if (catalog.ValueKind != JsonValueKind.Array) throw new JsonException("Antimatter model catalog has no data array.");
        foreach (var item in catalog.EnumerateArray())
        {
            var id = String(item, "id");
            if (string.IsNullOrWhiteSpace(id)) continue;
            var pricing = ProviderPrices.ReadPair(Property(item, "pricing"), "input", "output");
            yield return new()
            {
                Id = id, Type = "language", Name = String(item, "name") ?? String(item, "display_name") ?? id,
                OwnedBy = String(item, "owned_by") ?? Id, Description = String(item, "description"),
                Created = Property(item, "created") is { ValueKind: JsonValueKind.Number } created && created.TryGetInt64(out var timestamp) ? timestamp : null,
                ContextWindow = Number(item, "context_length"), MaxTokens = Number(item, "max_output"),
                Pricing = pricing is null ? null : new() { Input = pricing.Input / 1_000_000m, Output = pricing.Output / 1_000_000m }
            };
        }
    }

    private static int? Number(JsonElement item, string name)
        => Property(item, name) is { ValueKind: JsonValueKind.Number } value && value.TryGetInt32(out var number) ? number : null;
}
