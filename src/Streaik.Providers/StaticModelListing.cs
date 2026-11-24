using System.Runtime.CompilerServices;
using System.Text.Json;
using Streaik.Catalog;
using Streaik.OpenAI.Models;

namespace Streaik.Providers;

internal static class StaticModelListing
{
    public static async IAsyncEnumerable<Model> ListAsync(string providerId, string? apiKey, [EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(apiKey)) yield break;
        ct.ThrowIfCancellationRequested();
        var list = ProviderCatalog.ReadModels(providerId)?.Deserialize<ModelResponse>(MediaHttp.Json)
            ?? throw new InvalidOperationException($"Missing model catalog for '{providerId}'.");
        var prefix = $"{providerId}/";
        await Task.CompletedTask;
        foreach (var model in list.Data)
        {
            ct.ThrowIfCancellationRequested();
            if (!model.Id.StartsWith(prefix, StringComparison.Ordinal) || model.Id.Length == prefix.Length)
                throw new InvalidOperationException($"Invalid model ID in the '{providerId}' catalog.");
            model.Id = model.Id[prefix.Length..];
            yield return model;
        }
    }
}
