using System.Runtime.CompilerServices;
using System.Text.Json;
using Streaik.OpenAI.Models;

namespace Streaik.Providers;

/// <summary>Provider-specific image usage kept outside the serialized OpenAI image response.</summary>
public static class ImageProviderUsage
{
    private sealed record Usage(JsonElement Value);
    private static readonly ConditionalWeakTable<OpenAIImagesResponse, Usage> Data = new();

    internal static OpenAIImagesResponse Read(JsonElement body, JsonSerializerOptions json)
    {
        var response = body.Deserialize<OpenAIImagesResponse>(json) ?? throw new JsonException("Missing image response.");
        if (body.TryGetProperty("usage", out var usage) && usage.ValueKind == JsonValueKind.Object)
            Data.Add(response, new(usage));
        return response;
    }

    public static JsonElement Get(OpenAIImagesResponse response)
        => Data.TryGetValue(response, out var usage) ? usage.Value : default;
}
