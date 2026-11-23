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
    private sealed class Together(IHttpClientFactory clients, IProviderCredentials credentials) : CompatibleChatProvider(clients, credentials, "together");
    private sealed class Cerebras(IHttpClientFactory clients, IProviderCredentials credentials) : CompatibleChatProvider(clients, credentials, "cerebras");
}

internal abstract class CompatibleChatProvider(IHttpClientFactory clients, IProviderCredentials credentials, string id, string chatPrefix = "")
    : IChatCompletionsProvider, IModelListingProvider
{
    protected HttpClient CreateClient() => clients.CreateClient(id);
    protected string ApiKey() => RequiredKey(credentials, id);
    protected string? ConfiguredApiKey() => credentials.GetApiKey(id);

    public virtual async Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var auth = Bearer(ApiKey());
        using var client = CreateClient();
        if (request.Store == false) request.Store = null; // These endpoints do not define OpenAI's store flag.
        return await NativeHttp.PostAsync<ChatCompletionOptions, ChatCompletion>(client, chatPrefix + "chat/completions", request, auth, headers, Json, ct);
    }

    public virtual async IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var auth = Bearer(ApiKey());
        using var client = CreateClient();
        if (request.Store == false) request.Store = null;
        await foreach (var part in NativeHttp.StreamAsync<ChatCompletionOptions, ChatCompletionUpdate>(client, chatPrefix + "chat/completions", request, auth, headers, Json, _ => false, ct))
        {
            // Groq supplies streaming usage in an extension instead of the standard field.
            if (id == "groq" && part.Usage is null && part.AdditionalProperties?.TryGetValue("x_groq", out var groq) == true
                && Property(groq, "usage") is { ValueKind: JsonValueKind.Object } usage)
                part.Usage = usage.Deserialize<ChatCompletionUsage>(Json);
            yield return part;
        }
    }

    public virtual async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var key = credentials.GetApiKey(id);
        if (string.IsNullOrWhiteSpace(key)) yield break;
        using var client = CreateClient();
        var result = await NativeHttp.GetAsync<JsonElement>(client, chatPrefix + "models", Bearer(key), ct);
        // Together uses a bare array; the others use the OpenAI-style data envelope.
        foreach (var item in Array(result.ValueKind == JsonValueKind.Array ? result : Property(result, "data")))
        {
            if (String(item, "id") is not { Length: > 0 } model || Property(item, "active").ValueKind == JsonValueKind.False) continue;
            if (id == "google" && model.StartsWith("models/", StringComparison.Ordinal)) model = model[7..];
            var live = id == "google" && GoogleGeminiProvider.IsLiveModel(model)
                || id == "xai" && model == XaiProvider.RealtimeModel;
            var type = String(item, "type");
            var context = Property(item, "context_length");
            if (context.ValueKind != JsonValueKind.Number) context = Property(item, "context_window");
            yield return new Model { Id = model, Name = String(item, "display_name") ?? String(item, "name") ?? model,
                Type = type ?? (live ? "audio" : "model"), OwnedBy = String(item, "owned_by") ?? String(item, "organization") ?? id,
                Tags = Array(Property(item, "tags")).Where(tag => tag.ValueKind == JsonValueKind.String)
                    .Select(tag => tag.GetString()!).Concat(live ? ["realtime"] : []).ToArray(),
                ContextWindow = context.ValueKind == JsonValueKind.Number && context.TryGetInt32(out var size) ? size : null };
        }
    }

    private static Action<HttpRequestHeaders> Bearer(string key) => headers => headers.Authorization = new("Bearer", key);
}
