using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace Streaik.Providers;

public sealed record ProviderRegistration(string Id, IReadOnlyList<string> AdditionalHeaders);

public static class ProviderServiceCollectionExtensions
{
    public static IServiceCollection AddProvider<T>(this IServiceCollection services, string id, params string[] additionalHeaders) where T : class
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        if (!id.All(c => char.IsAsciiLetterLower(c) || char.IsAsciiDigit(c) || c is '-' or '_'))
            throw new ArgumentException("Provider IDs must be lowercase URL segments.", nameof(id));
        if (services.Any(s => s.ServiceType == typeof(ProviderRegistration)
            && s.ImplementationInstance is ProviderRegistration registration && registration.Id == id))
            throw new InvalidOperationException($"Provider '{id}' is already registered.");

        var registration = new ProviderRegistration(id, Array.AsReadOnly(additionalHeaders.ToArray()));
        services.AddSingleton(registration);
        services.AddKeyedSingleton(id, registration);
        services.TryAddScoped<T>();
        Register<IModelListingProvider>();
        Register<IChatCompletionsProvider>();
        Register<IChatBridgeProvider>();
        Register<INativeChatOptionsProvider>();
        Register<IResponsesProvider>();
        Register<IMessagesProvider>();
        Register<IDecisionsProvider>();
        Register<IImageProvider>();
        Register<IStreamingImagesProvider>();
        Register<IVideoProvider>();
        Register<ISpeechProvider>();
        Register<IStreamingSpeechProvider>();
        Register<ITranscriptionProvider>();
        Register<IStreamingTranscriptionProvider>();
        Register<INativeAudioTranscriptionProvider>();
        Register<IRealtimeTokenProvider>();
        Register<IRealtimeSessionProvider>();
        Register<IOpenAIRealtimeProvider>();
        Register<IOpenAILiveSessionProvider>();
        Register<IEmbeddingsProvider>();
        Register<IEmbeddingMetadataProvider>();
        Register<IRerankingProvider>();
        Register<IContainerFilesProvider>();
        Register<ISkillReadProvider>();
        Register<ISkillUploadProvider>();
        return services;

        void Register<TCapability>() where TCapability : class
        {
            if (typeof(TCapability).IsAssignableFrom(typeof(T)))
                services.AddKeyedScoped<TCapability>(id, (sp, _) => (TCapability)(object)sp.GetRequiredService<T>());
        }
    }

    internal static void AddClient(IServiceCollection services, string id, string address)
    {
        services.TryAddScoped<IProviderCredentials, ConfigurationProviderCredentials>();
        services.AddHttpClient(id, client => client.BaseAddress = new Uri(address))
            .ConfigurePrimaryHttpMessageHandler(() => new SocketsHttpHandler { UseCookies = false, AllowAutoRedirect = false });
    }

    private sealed class ConfigurationProviderCredentials(IConfiguration configuration) : IProviderCredentials
    {
        public string? GetApiKey(string providerId) => configuration[$"Providers:{providerId}:ApiKey"]
            ?? configuration[$"{providerId.ToUpperInvariant()}_API_KEY"];
    }
}
