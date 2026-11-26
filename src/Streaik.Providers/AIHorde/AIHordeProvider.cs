using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class AIHordeProviderRegistration
{
    public static IServiceCollection AddAIHorde(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AIHordeProvider.Id, "https://oai.aihorde.net/v1/");
        ProviderServiceCollectionExtensions.AddClient(services, AIHordeProvider.ImageClient, "https://aihorde.net/api/v2/");
        ProviderServiceCollectionExtensions.AddClient(services, AIHordeProvider.AssetClient, "https://aihorde.net/");
        return services.AddProvider<AIHordeProvider>(AIHordeProvider.Id);
    }
}

internal sealed partial class AIHordeProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), INativeChatOptionsProvider, IImageModelRoutingProvider
{
    internal const string Id = "aihorde";
    internal const string ImageClient = "aihorde-images";
    internal const string AssetClient = "aihorde-assets";
    private const string ImageModel = "aihorde-image";

    private static bool IsImageModel(string model) => model.Equals(ImageModel, StringComparison.OrdinalIgnoreCase);

    public Task<bool> IsImageModelAsync(string model, CancellationToken ct) => Task.FromResult(IsImageModel(model));

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        var body = await NativeHttp.GetAsync<JsonElement>(client, "models", _ => { }, ct);
        foreach (var item in Array(Property(body, "data")))
        {
            if (String(item, "id") is not { Length: > 0 } model || IsImageModel(model)) continue;
            yield return new() { Id = model, Name = String(item, "clean_name") ?? model, Type = "model",
                OwnedBy = String(item, "backend") ?? Id };
        }
        yield return new() { Id = ImageModel, Name = ImageModel, Type = "image", OwnedBy = "AI Horde",
            Description = "AI Horde image generation and editing." };
    }
}
