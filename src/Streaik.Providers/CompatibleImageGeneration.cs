using System.Net.Http.Json;
using System.Text.Json;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal static class CompatibleImageGeneration
{
    public static async Task<OpenAIImagesResponse> GenerateAsync(HttpClient client, string provider, string key,
        OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct,
        string path = "v1/images/generations")
    {
        var payload = ToObject(request);
        payload.Remove("stream");
        payload.Remove("partial_images");
        using var response = await SendAsync(client, HttpMethod.Post, path, JsonContent.Create(payload, options: Json),
            "Bearer", key, headers, ct);
        var body = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        var result = body.Deserialize<OpenAIImagesResponse>(Json) ?? throw new JsonException("Missing image response.");
        if (result.Data is not { Count: > 0 } || Array(Property(body, "data")).Any(image =>
            string.IsNullOrWhiteSpace(String(image, "b64_json")) && string.IsNullOrWhiteSpace(String(image, "url"))))
            throw new JsonException("The provider returned no usable images.");

        result.OutputFormat ??= request.OutputFormat ?? "png";
        if (Property(body, "created").ValueKind == JsonValueKind.Undefined)
            result.Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var metadata = (result.AdditionalProperties ??= new(StringComparer.Ordinal));
        // Keep native fields (including per-image diagnostics) without duplicating large image bodies.
        var raw = ToObject(body);
        foreach (var item in raw["data"]!.AsArray())
        {
            item!.AsObject().Remove("b64_json");
            item.AsObject().Remove("url");
        }
        metadata.TryAdd("providerMetadata", JsonSerializer.SerializeToElement(new Dictionary<string, object> { [provider] = raw }, Json));
        return result;
    }
}
