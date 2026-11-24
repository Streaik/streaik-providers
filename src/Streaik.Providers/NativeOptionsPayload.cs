using System.Text.Json;

namespace Streaik.Providers;

internal static class NativeOptionsPayload
{
    private static readonly HashSet<string> Controls = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "stream", "store", "background", "previous_response_id", "conversation", "input", "messages",
        "authorization", "api_key", "apiKey", "x-api-key", "headers"
    };

    public static Dictionary<string, JsonElement> Build<T>(T request, string provider, JsonSerializerOptions json, bool? stream = null)
    {
        var source = JsonSerializer.SerializeToElement(request, json);
        var typed = json.GetTypeInfo(typeof(T)).Properties.Where(property => !property.IsExtensionData)
            .Select(property => property.Name).ToHashSet(StringComparer.Ordinal);
        var reserved = new HashSet<string>(typed, StringComparer.OrdinalIgnoreCase);
        reserved.UnionWith(Controls);
        reserved.Add(provider);
        var body = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        if (source.TryGetProperty(provider, out var options))
        {
            if (options.ValueKind != JsonValueKind.Object) throw new ProviderRequestException($"{provider} options must be an object.");
            foreach (var property in options.EnumerateObject())
                if (!reserved.Contains(property.Name)) body[property.Name] = property.Value;
        }
        foreach (var property in source.EnumerateObject())
            if (typed.Contains(property.Name) || !reserved.Contains(property.Name)) body[property.Name] = property.Value;
        if (stream is { } enabled) body["stream"] = JsonSerializer.SerializeToElement(enabled);
        return body;
    }
}
