using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class CompatibleChatProviderRegistration
{
    public static IServiceCollection AddDeepSeek(this IServiceCollection services)
        => Add<DeepSeek>(services, "deepseek", "https://api.deepseek.com/");

    public static IServiceCollection AddGroq(this IServiceCollection services)
        => Add<Groq>(services, "groq", "https://api.groq.com/openai/v1/");

    public static IServiceCollection AddTogether(this IServiceCollection services)
        => Add<Together>(services, "together", "https://api.together.ai/v1/");

    public static IServiceCollection AddCerebras(this IServiceCollection services)
        => Add<Cerebras>(services, "cerebras", "https://api.cerebras.ai/v1/");

    public static IServiceCollection AddMoonshot(this IServiceCollection services)
        => Add<Moonshot>(services, "moonshot", "https://api.moonshot.ai/v1/");

    public static IServiceCollection AddChutes(this IServiceCollection services)
        => Add<Chutes>(services, "chutes", "https://llm.chutes.ai/v1/");

    public static IServiceCollection AddCloudRift(this IServiceCollection services)
        => Add<CloudRift>(services, "cloudrift", "https://inference.cloudrift.ai/v1/");

    public static IServiceCollection AddCrusoe(this IServiceCollection services)
        => Add<Crusoe>(services, "crusoe", "https://api.inference.crusoecloud.com/v1/");

    public static IServiceCollection AddArkRoute(this IServiceCollection services)
        => Add<ArkRoute>(services, "arkroute", "https://api.ark-route.com/v1/");

    public static IServiceCollection AddBergetAI(this IServiceCollection services)
        => Add<BergetAIProvider>(services, "bergetai", "https://api.berget.ai/v1/");

    public static IServiceCollection AddGoogle(this IServiceCollection services)
        => Add<GoogleGeminiProvider>(services, "google", "https://generativelanguage.googleapis.com/");

    public static IServiceCollection AddXai(this IServiceCollection services)
        => Add<XaiProvider>(services, "xai", "https://api.x.ai/");

    private static IServiceCollection Add<T>(IServiceCollection services, string id, string address) where T : CompatibleChatProvider
    {
        ProviderServiceCollectionExtensions.AddClient(services, id, address);
        return services.AddProvider<T>(id);
    }

    private sealed class DeepSeek(IHttpClientFactory clients, IProviderCredentials credentials) : CompatibleChatProvider(clients, credentials, "deepseek");
    private sealed class Groq(IHttpClientFactory clients, IProviderCredentials credentials) : CompatibleChatProvider(clients, credentials, "groq");
    private sealed class Together(IHttpClientFactory clients, IProviderCredentials credentials)
        : CompatibleChatProvider(clients, credentials, "together"), INativeChatOptionsProvider;
    private sealed class Cerebras(IHttpClientFactory clients, IProviderCredentials credentials)
        : CompatibleChatProvider(clients, credentials, "cerebras"), INativeChatOptionsProvider;
    private sealed class ArkRoute(IHttpClientFactory clients, IProviderCredentials credentials)
        : CompatibleChatProvider(clients, credentials, "arkroute"), INativeChatOptionsProvider;
    private sealed class CloudRift(IHttpClientFactory clients, IProviderCredentials credentials)
        : CompatibleChatProvider(clients, credentials, "cloudrift"), INativeChatOptionsProvider;
    private sealed class Chutes(IHttpClientFactory clients, IProviderCredentials credentials)
        : CompatibleChatProvider(clients, credentials, "chutes"), INativeChatOptionsProvider;
    private sealed class Crusoe(IHttpClientFactory clients, IProviderCredentials credentials)
        : CompatibleChatProvider(clients, credentials, "crusoe"), INativeChatOptionsProvider
    {
        protected override void PrepareCompatibleRequest(ChatCompletionOptions request)
        {
            request.Store = null;
            if (request.Tools?.Any() == false) request.Tools = null!;
        }

        protected override ModelPricing? Pricing(JsonElement model)
        {
            var pricing = Property(model, "pricing");
            if (ProviderPrices.ReadPair(pricing, "prompt", "completion") is not { } price) return null;
            price.Input /= 1_000_000m;
            price.Output /= 1_000_000m;
            if (ProviderPrices.TokenPrice(Property(pricing, "input_cache_reads")) is >= 0 and var cacheRead)
                price.InputCacheRead = cacheRead / 1_000_000m;
            return price;
        }
    }

    private sealed class Moonshot(IHttpClientFactory clients, IProviderCredentials credentials)
        : CompatibleChatProvider(clients, credentials, "moonshot"), INativeChatOptionsProvider
    {
        protected override ModelPricing? Pricing(JsonElement model)
            => Property(model, "pricing") is { ValueKind: JsonValueKind.Object } pricing
                && Property(pricing, "input").ValueKind is JsonValueKind.Number or JsonValueKind.String
                && Property(pricing, "output").ValueKind is JsonValueKind.Number or JsonValueKind.String
                    ? pricing.Deserialize<ModelPricing>(Json) : null;
    }
}

internal abstract class CompatibleChatProvider(IHttpClientFactory clients, IProviderCredentials credentials, string id, string chatPrefix = "")
    : IChatCompletionsProvider, IModelListingProvider
{
    protected HttpClient CreateClient(string? name = null) => clients.CreateClient(name ?? id);
    protected string ApiKey() => RequiredKey(credentials, id);
    protected string? ConfiguredApiKey() => credentials.GetApiKey(id);
    protected string ProviderId => id;
    protected string CompatiblePath(string endpoint) => chatPrefix + endpoint;

    public virtual async Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var auth = ChatAuth();
        using var client = CreateClient();
        PrepareCompatibleRequest(request);
        return await NativeHttp.PostAsync<object, ChatCompletion>(client, CompatiblePath("chat/completions"), ChatBody(request, false), auth, headers, Json, ct);
    }

    public virtual async IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var auth = ChatAuth();
        using var client = CreateClient();
        PrepareCompatibleRequest(request);
        await foreach (var part in NativeHttp.StreamAsync<object, ChatCompletionUpdate>(client, CompatiblePath("chat/completions"), ChatBody(request, true), auth, headers, Json, _ => false, ct))
        {
            // Groq supplies streaming usage in an extension instead of the standard field.
            if (id == "groq" && part.Usage is null && part.AdditionalProperties?.TryGetValue("x_groq", out var groq) == true
                && Property(groq, "usage") is { ValueKind: JsonValueKind.Object } usage)
                part.Usage = usage.Deserialize<ChatCompletionUsage>(Json);
            yield return part;
        }
    }

    private object ChatBody(ChatCompletionOptions request, bool stream)
        => this is INativeChatOptionsProvider && request.AdditionalProperties?.ContainsKey(id) == true
            ? NativeOptionsPayload.Build(request, id, Json, stream) : request;

    public virtual async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var key = credentials.GetApiKey(id);
        if (string.IsNullOrWhiteSpace(key)) yield break;
        var result = await FetchModelsAsync(key, ct);
        // Together uses a bare array; the others use the OpenAI-style data envelope.
        foreach (var item in Array(result.ValueKind == JsonValueKind.Array ? result : Property(result, "data")))
        {
            if (String(item, "id") is not { Length: > 0 } model || Property(item, "active").ValueKind == JsonValueKind.False) continue;
            if (id == "google")
            {
                // Return the native ID; the gateway adds its own routing prefix.
                if (model.StartsWith("models/", StringComparison.Ordinal)) model = model[7..];
                if (model.StartsWith("google/", StringComparison.Ordinal)) model = model[7..];
                if (model.Length == 0) continue;
            }
            var live = id == "google" && GoogleGeminiProvider.IsLiveModel(model)
                || id == "xai" && model == XaiProvider.RealtimeModel;
            var type = String(item, "type");
            var context = Property(item, "context_length");
            if (context.ValueKind != JsonValueKind.Number) context = Property(item, "context_window");
            yield return new Model { Id = model, Name = String(item, "display_name") ?? String(item, "name") ?? model,
                Type = type ?? (live ? "audio" : "model"), OwnedBy = String(item, "owned_by") ?? String(item, "organization") ?? id,
                Description = String(item, "description"),
                Tags = live ? ["realtime"] : null,
                ContextWindow = context.ValueKind == JsonValueKind.Number && context.TryGetInt32(out var size) ? size : null,
                Pricing = Pricing(item) };
        }
    }

    protected virtual async Task<JsonElement> FetchModelsAsync(string key, CancellationToken ct)
    {
        using var client = CreateClient();
        return await NativeHttp.GetAsync<JsonElement>(client, CompatiblePath("models"), Bearer(key), ct);
    }

    protected virtual void PrepareCompatibleRequest(ChatCompletionOptions request)
    {
        if (request.Store == false) request.Store = null; // These endpoints do not define OpenAI's store flag.
    }

    protected virtual ModelPricing? Pricing(JsonElement model) => null;
    protected virtual Action<HttpRequestHeaders> ChatAuth() => Bearer(ApiKey());
    protected static Action<HttpRequestHeaders> Bearer(string key) => headers => headers.Authorization = new("Bearer", key);
}
