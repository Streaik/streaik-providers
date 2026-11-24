using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class NomicProviderRegistration
{
    public static IServiceCollection AddNomic(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, NomicProvider.Id, "https://api-atlas.nomic.ai/");
        return services.AddProvider<NomicProvider>(NomicProvider.Id);
    }
}

internal sealed class NomicProvider(IHttpClientFactory clients, IProviderCredentials credentials) : IModelListingProvider, IEmbeddingMetadataProvider
{
    internal const string Id = "nomic";
    private static readonly string[] Models = ["nomic-embed-text-v1.5", "nomic-embed-text-v1"];

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        foreach (var model in Models)
        {
            ct.ThrowIfCancellationRequested();
            yield return new Model { Id = model, Name = model, Type = "embedding", OwnedBy = "Nomic" };
        }
        await Task.CompletedTask;
    }

    public async Task<OpenAIEmbeddingResponse> EmbedAsync(OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => (await EmbedWithMetadataAsync(request, headers, ct)).Result;

    public async Task<(OpenAIEmbeddingResponse Result, JsonElement Body, Dictionary<string, string?> Headers)> EmbedWithMetadataAsync(
        OpenAIEmbeddingRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var encoding = request.EncodingFormat?.ToLowerInvariant() ?? "float";
        if (encoding is not ("float" or "base64") || request.Dimensions is <= 0 || request.User is not null)
            throw new ProviderRequestException("Nomic supports float or base64 embeddings, positive dimensions, and no user field.");
        var texts = request.Input.ValueKind switch
        {
            JsonValueKind.String => [request.Input.GetString()!],
            JsonValueKind.Array when request.Input.EnumerateArray().All(item => item.ValueKind == JsonValueKind.String)
                => request.Input.EnumerateArray().Select(item => item.GetString()!).ToArray(),
            _ => throw new ProviderRequestException("Nomic requires text strings, not token IDs.")
        };
        if (texts.Length == 0) throw new ProviderRequestException("Nomic requires at least one input string.");

        var payload = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (var (name, value) in request.AdditionalProperties ?? [])
            if (name is not ("model" or "texts" or "input" or "encoding_format" or "user" or "dimensions")) payload[name] = value;
        payload["model"] = JsonSerializer.SerializeToElement(request.Model);
        payload["texts"] = JsonSerializer.SerializeToElement(texts);
        if (request.Dimensions is { } dimensions) payload["dimensionality"] = JsonSerializer.SerializeToElement(dimensions);

        var key = RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, "v1/embedding/text", payload,
            h => h.Authorization = new("Bearer", key), headers, Json, ct);
        var vectors = Property(body, "embeddings");
        if (vectors.ValueKind != JsonValueKind.Array || vectors.GetArrayLength() != texts.Length)
            throw new JsonException("Nomic did not return one embedding per input.");
        var data = vectors.EnumerateArray().Select((vector, index) => new OpenAIEmbeddingData
        {
            Index = index, Embedding = ReadVector(vector, encoding == "base64")
        }).ToArray();
        var usage = Property(body, "usage");
        return (new() { Model = String(body, "model") ?? request.Model, Data = data,
            Usage = usage.ValueKind == JsonValueKind.Object ? usage.Deserialize<OpenAIEmbeddingUsage>(Json)! : new() },
            body, NativeHttp.SafeResponseMetadataHeaders(upstream));
    }

    private static JsonElement ReadVector(JsonElement vector, bool base64)
    {
        if (vector.ValueKind != JsonValueKind.Array || vector.GetArrayLength() == 0)
            throw new JsonException("Nomic returned an empty or invalid embedding vector.");
        var bytes = base64 ? new byte[checked(vector.GetArrayLength() * sizeof(float))] : null;
        var offset = 0;
        foreach (var component in vector.EnumerateArray())
        {
            if (component.ValueKind != JsonValueKind.Number || !component.TryGetSingle(out var value) || !float.IsFinite(value))
                throw new JsonException("Nomic returned a non-finite or invalid embedding component.");
            if (bytes is null) continue;
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(offset), value);
            offset += sizeof(float);
        }
        return bytes is null ? vector : JsonSerializer.SerializeToElement(Convert.ToBase64String(bytes));
    }
}
