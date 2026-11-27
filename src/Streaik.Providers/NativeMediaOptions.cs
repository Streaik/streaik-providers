using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal static class NativeMediaOptions
{
    private static readonly HashSet<string> TransportFields = new(StringComparer.OrdinalIgnoreCase)
    {
        "providerOptions", "providerMetadata", "authorization", "api-key", "api_key", "apiKey", "x-api-key", "headers"
    };

    internal static Dictionary<string, JsonElement> ReadScoped(string id, IReadOnlyDictionary<string, JsonElement>? options,
        IReadOnlySet<string> reserved)
        => Read(id, options?.TryGetValue(id, out var native) == true ? new Dictionary<string, JsonElement> { [id] = native } : null, reserved);

    internal static Dictionary<string, JsonElement> Read(string id, IReadOnlyDictionary<string, JsonElement>? options,
        IReadOnlySet<string> reserved)
    {
        var result = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        if (options is not null)
            foreach (var (name, value) in options) Add(name, value);
        if (options?.TryGetValue(id, out var scoped) == true) Merge(scoped);
        if (options?.TryGetValue("providerOptions", out var envelope) == true)
        {
            if (envelope.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("providerOptions must be an object.");
            if (envelope.TryGetProperty(id, out var native)) Merge(native);
        }
        return result;

        void Add(string name, JsonElement value)
        {
            if (!TransportFields.Contains(name) && !reserved.Contains(name) && !name.Equals(id, StringComparison.OrdinalIgnoreCase))
                result[name] = value;
        }

        void Merge(JsonElement value)
        {
            if (value.ValueKind != JsonValueKind.Object) throw new ProviderRequestException($"{id} options must be an object.");
            foreach (var property in value.EnumerateObject()) Add(property.Name, property.Value);
        }
    }

    internal static Dictionary<string, JsonElement> Metadata<T>(string id, T value)
        => new() { [id] = JsonSerializer.SerializeToElement(value, Json) };

    internal static JsonElement CompletedImage(OpenAIImagesResponse result, bool edit)
    {
        if (result.Data is not [{ B64Json: { Length: > 0 } image }])
            throw new JsonException("Image streaming requires a single base64 output.");
        OpenAIImageStreamEventBase completed = edit
            ? new OpenAIImageEditCompleted { B64Json = image, Usage = result.Usage }
            : new OpenAIImageGenerationCompleted { B64Json = image, Usage = result.Usage };
        completed.CreatedAt = result.Created;
        completed.OutputFormat = result.OutputFormat;
        completed.Background = result.Background;
        completed.Quality = result.Quality;
        completed.Size = result.Size;
        var wire = (JsonObject)JsonSerializer.SerializeToNode(completed, completed.GetType(), Json)!;
        if (result.AdditionalProperties?.TryGetValue("providerMetadata", out var metadata) == true)
            wire["providerMetadata"] = Copy(metadata);
        return JsonSerializer.SerializeToElement(wire, Json);
    }
}
