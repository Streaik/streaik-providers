using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.CanopyWave;

public static class CanopyWaveRegistration
{
    public static IServiceCollection AddCanopyWave(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, CanopyWaveProvider.Id, "https://inference.canopywave.io/v1/");
        return services.AddProvider<CanopyWaveProvider>(CanopyWaveProvider.Id);
    }
}

internal sealed class CanopyWaveProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id)
{
    internal const string Id = "canopywave";

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (ConfiguredApiKey() is not { } key || string.IsNullOrWhiteSpace(key)) yield break;
        using var client = CreateClient();
        var result = await NativeHttp.GetAsync<JsonElement>(client, "models", headers => headers.Authorization = new("Bearer", key), ct);
        foreach (var item in Array(Property(result, "data")))
        {
            if (String(item, "id") is not { Length: > 0 } model) continue;
            var input = Price(Property(item, "input_token_price_per_m"));
            var output = Price(Property(item, "output_token_price_per_m"));
            yield return new Model
            {
                Id = model,
                Name = String(item, "display_name") ?? model,
                OwnedBy = String(item, "owned_by") ?? Id,
                Type = "language",
                Description = String(item, "description"),
                ContextWindow = Integer(Property(item, "context_size")),
                MaxTokens = Integer(Property(item, "max_output_tokens")),
                Pricing = input is >= 0 && output is >= 0 ? new ModelPricing { Input = input.Value, Output = output.Value } : null
            };
        }
    }

    private static int? Integer(JsonElement value) => value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out var n) ? n : null;

    private static decimal? Price(JsonElement value)
        => decimal.TryParse(value.ValueKind == JsonValueKind.String ? value.GetString() : value.ToString(), NumberStyles.Float,
            CultureInfo.InvariantCulture, out var price) ? price / 1_000_000m : null;
}
