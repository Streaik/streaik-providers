using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class CreoraProviderRegistration
{
    public static IServiceCollection AddCreora(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, CreoraProvider.Id, "https://creora.ai/api/v1/");
        return services.AddProvider<CreoraProvider>(CreoraProvider.Id);
    }
}

internal sealed partial class CreoraProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), INativeChatOptionsProvider, IImageModelRoutingProvider, IVideoProvider
{
    internal const string Id = "creora";

    public override async Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, "chat/completions", ChatPayload(request, false), Bearer(ApiKey()), headers, Json, ct);
        var completion = body.Deserialize<ChatCompletion>(Json) ?? throw new JsonException("Missing Creora chat response.");
        var extensions = completion.AdditionalProperties ??= [];
        extensions.TryAdd("providerMetadata", JsonSerializer.SerializeToElement(new Dictionary<string, object>
        {
            [Id] = new { headers = NativeHttp.SafeResponseMetadataHeaders(upstream), response = body }
        }, Json));
        return completion;
    }

    public override async IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        await foreach (var part in NativeHttp.StreamAsync<Dictionary<string, JsonElement>, ChatCompletionUpdate>(client, "chat/completions",
            ChatPayload(request, true), Bearer(ApiKey()), headers, Json, part => part.AdditionalProperties?.ContainsKey("error") == true, ct))
            yield return part;
    }

    private Dictionary<string, JsonElement> ChatPayload(ChatCompletionOptions request, bool stream)
    {
        PrepareCompatibleRequest(request);
        var body = NativeOptionsPayload.Build(request, Id, Json, stream);
        if (request.Tools?.Any() != true)
        {
            body.Remove("tools");
            if (request.AdditionalProperties?.TryGetValue(Id, out var options) == true && Property(options, "tools") is { ValueKind: JsonValueKind.Array } tools)
                body["tools"] = tools;
        }
        return body;
    }

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (ConfiguredApiKey() is not { } key || string.IsNullOrWhiteSpace(key)) yield break;
        var catalog = await FetchModelsAsync(key, ct);
        foreach (var item in Array(Property(catalog, "data")))
        {
            ct.ThrowIfCancellationRequested();
            if (String(item, "id") is not { Length: > 0 } id) continue;
            if (id.StartsWith(Id + "/", StringComparison.Ordinal)) id = id[(Id.Length + 1)..];
            if (id.Length == 0) continue;
            var model = item.Deserialize<Model>(Json) ?? throw new JsonException("Missing Creora model.");
            model.Id = id;
            if (string.IsNullOrWhiteSpace(model.Name)) model.Name = id;
            model.OwnedBy ??= Id;
            model.Type = model.Type switch
            {
                "chat" or "text" => "language",
                null or "" => id switch
                {
                    "creora-chat" or "creora-network" => "language",
                    "creora-image" => "image",
                    "creora-video" => "video",
                    _ => "model"
                },
                var type => type
            };
            yield return model;
        }
    }

    public async Task<bool> IsImageModelAsync(string model, CancellationToken ct)
    {
        if (model == "creora-image") return true;
        if (model is "creora-chat" or "creora-network" or "creora-video") return false;
        await foreach (var entry in ListModelsAsync(ct))
            if (entry.Id == model) return entry.Type == "image";
        return false;
    }
}
