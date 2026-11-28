using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.DependencyInjection;
using Streaik.Messages;
using Streaik.OpenAI.Models;

namespace Streaik.Providers.Anthropic;

public static class AnthropicRegistration
{
    public static IServiceCollection AddAnthropic(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AnthropicProvider.Id, "https://api.anthropic.com/");
        return services.AddProvider<AnthropicProvider>(AnthropicProvider.Id);
    }
}

public sealed class AnthropicProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IMessagesProvider, IModelListingProvider
{
    public const string Id = "anthropic";

    public async Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var key = RequireKey();
        var version = RequireVersion(request, headers);
        using var client = clients.CreateClient(Id);
        return await NativeHttp.PostAsync<MessagesRequest, MessagesResponse>(client, "v1/messages", request, h =>
            Authenticate(h, key, version), headers, MessagesJson.Default, ct);
    }

    public async IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        var key = RequireKey();
        var version = RequireVersion(request, headers);
        using var client = clients.CreateClient(Id);
        await foreach (var part in NativeHttp.StreamAsync<MessagesRequest, MessageStreamPart>(client, "v1/messages", request,
            h => Authenticate(h, key, version), headers, MessagesJson.Default,
            part => part.Type is "message_stop" or "error", ct)) yield return part;
    }

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var key = credentials.GetApiKey(Id);
        if (string.IsNullOrWhiteSpace(key)) yield break;
        using var client = clients.CreateClient(Id);
        var path = "v1/models?limit=100";
        var cursors = new HashSet<string>(StringComparer.Ordinal);
        while (true)
        {
            var page = await NativeHttp.GetAsync<ModelPage>(client, path, h => Authenticate(h, key, "2023-06-01"), ct, ProviderModelJson.Options);
            foreach (var item in page.Data)
            {
                if (item.Deserialize<Model>(ProviderModelJson.Options) is not { } model) continue;
                model.Created ??= ProviderModelJson.ReadCreated(item);
                yield return model;
            }

            if (!page.HasMore) yield break;
            if (page.LastId is not { Length: > 0 } cursor || !cursors.Add(cursor))
                throw new HttpRequestException("Invalid Anthropic model list cursor.");
            path = $"v1/models?limit=100&after_id={Uri.EscapeDataString(cursor)}";
        }
    }

    private string RequireKey() => credentials.GetApiKey(Id) is { } key && !string.IsNullOrWhiteSpace(key)
        ? key : throw new ProviderCredentialException(Id);

    private sealed record ModelPage(List<JsonElement> Data, [property: JsonPropertyName("has_more")] bool HasMore,
        [property: JsonPropertyName("last_id")] string? LastId);

    private static string RequireVersion(MessagesRequest request, IReadOnlyDictionary<string, string> headers)
    {
        if (request.MaxTokens <= 0) throw new ProviderRequestException("Anthropic requires a positive max_tokens value.");
        var version = headers.FirstOrDefault(header => header.Key.Equals("anthropic-version", StringComparison.OrdinalIgnoreCase)).Value;
        return !string.IsNullOrWhiteSpace(version) ? version : throw new ProviderRequestException("Anthropic requires the anthropic-version header.");
    }

    private static void Authenticate(HttpRequestHeaders headers, string key, string version)
    {
        headers.Add("x-api-key", key);
        headers.Add("anthropic-version", version);
    }
}
