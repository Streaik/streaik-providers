using System.Buffers.Binary;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class HuggingFaceProviderRegistration
{
    public static IServiceCollection AddHuggingFace(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, HuggingFaceProvider.Id, "https://router.huggingface.co/");
        return services.AddProvider<HuggingFaceProvider>(HuggingFaceProvider.Id);
    }
}

internal sealed class HuggingFaceProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id, "v1/"), INativeChatOptionsProvider, IEmbeddingMetadataProvider
{
    internal const string Id = "huggingface";

    public async Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => (await EmbedWithMetadataAsync(request, headers, ct)).Result;

    public async Task<(OpenAIEmbeddingResponse Result, JsonElement Body, Dictionary<string, string?> Headers)> EmbedWithMetadataAsync(
        OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Dimensions is not null || request.User is not null)
            throw new ProviderRequestException("Hugging Face feature extraction does not support dimensions or user.", 422);
        var format = request.EncodingFormat?.ToLowerInvariant();
        if (format is not (null or "float" or "base64"))
            throw new ProviderRequestException("Expected float or base64 embedding encoding.");
        var count = request.Input.ValueKind switch
        {
            JsonValueKind.String => 1,
            JsonValueKind.Array when request.Input.EnumerateArray().All(item => item.ValueKind == JsonValueKind.String)
                => request.Input.GetArrayLength(),
            _ => 0
        };
        if (count == 0) throw new ProviderRequestException("Hugging Face feature extraction requires text or a non-empty array of texts.", 422);

        var segments = request.Model.Split('/');
        if (segments.Any(segment => string.IsNullOrWhiteSpace(segment) || segment is "." or ".." || segment.Contains('\\')))
            throw new ProviderRequestException("Invalid Hugging Face model ID.");
        var modelPath = string.Join('/', segments.Select(Uri.EscapeDataString));
        var payload = new Dictionary<string, JsonElement>(StringComparer.Ordinal) { ["inputs"] = request.Input };
        foreach (var (name, value) in request.AdditionalProperties ?? [])
            if (!name.Equals("inputs", StringComparison.OrdinalIgnoreCase)) payload[name] = value;

        using var client = CreateClient();
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, $"hf-inference/models/{modelPath}/pipeline/feature-extraction",
            payload, Bearer(ApiKey()), headers, Json, ct);
        if (body.ValueKind != JsonValueKind.Array || body.GetArrayLength() == 0)
            throw new JsonException("Expected Hugging Face feature-extraction embeddings.");
        var vectors = count == 1 && body[0].ValueKind == JsonValueKind.Number ? [body] : body.EnumerateArray().ToArray();
        if (vectors.Length != count) throw new JsonException("Expected one Hugging Face embedding per input.");
        var data = vectors.Select((vector, index) => new OpenAIEmbeddingData
        {
            Index = index, Embedding = ReadVector(vector, format == "base64")
        }).ToArray();
        return (new() { Model = request.Model, Data = data }, body, NativeHttp.SafeResponseMetadataHeaders(upstream));
    }

    private static JsonElement ReadVector(JsonElement vector, bool base64)
    {
        if (vector.ValueKind != JsonValueKind.Array || vector.GetArrayLength() == 0)
            throw new JsonException("Expected a non-empty Hugging Face embedding vector.");
        var bytes = base64 ? new byte[checked(vector.GetArrayLength() * sizeof(float))] : null;
        var offset = 0;
        foreach (var number in vector.EnumerateArray())
        {
            if (number.ValueKind != JsonValueKind.Number || !number.TryGetSingle(out var value) || !float.IsFinite(value))
                throw new JsonException("Expected finite numbers in the Hugging Face embedding vector.");
            if (bytes is not null) BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(offset, sizeof(float)), value);
            offset += sizeof(float);
        }
        return bytes is null ? vector : JsonSerializer.SerializeToElement(Convert.ToBase64String(bytes));
    }
}
