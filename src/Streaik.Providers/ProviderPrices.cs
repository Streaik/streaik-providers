using System.Globalization;
using System.Text.Json;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal static class ProviderPrices
{
    public static ModelPricing? ReadPair(JsonElement price, string input, string output)
        => TokenPrice(Property(price, input)) is >= 0 and var prompt && TokenPrice(Property(price, output)) is >= 0 and var completion
            ? new() { Input = prompt, Output = completion } : null;

    internal static decimal? TokenPrice(JsonElement price) => price switch
    {
        { ValueKind: JsonValueKind.Number } when price.TryGetDecimal(out var value) => value,
        { ValueKind: JsonValueKind.String } when decimal.TryParse(price.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var value) => value,
        _ => null
    };
}
