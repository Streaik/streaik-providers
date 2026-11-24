using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class DigitalOceanProviderRegistration
{
    public static IServiceCollection AddDigitalOcean(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, DigitalOceanProvider.Id, "https://inference.do-ai.run/");
        return services.AddProvider<DigitalOceanProvider>(DigitalOceanProvider.Id);
    }
}

internal sealed partial class DigitalOceanProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id, "v1/"), INativeChatOptionsProvider, IImageModelRoutingProvider, IDecisionsProvider
{
    internal const string Id = "digitalocean";

    private static bool IsImageModel(string model) => new[] { "flux", "image", "stable-diffusion", "sdxl", "sd3.5", "dall-e",
        "dalle", "imagen", "ideogram", "riverflow", "kandinsky", "dreamshaper", "bria", "seedream", "recraft", "hidream", "upscale" }
        .Any(marker => model.Contains(marker, StringComparison.OrdinalIgnoreCase));

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        await foreach (var model in base.ListModelsAsync(ct))
        {
            if (model.Type is not ("model" or "language" or "chat" or "image" or "decision")) continue;
            if (model.Type == "model")
            {
                if (new[] { "embed", "rerank", "whisper", "transcribe", "tts", "realtime", "video" }
                    .Any(marker => model.Id.Contains(marker, StringComparison.OrdinalIgnoreCase))) continue;
                model.Type = IsImageModel(model.Id) ? "image"
                    : model.Id.Contains("systemone", StringComparison.OrdinalIgnoreCase) ? "decision" : "language";
            }
            yield return model;
        }
    }

    public Task<bool> IsImageModelAsync(string model, CancellationToken ct) => Task.FromResult(IsImageModel(model));

    public override Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Tools?.Any() == false) request.Tools = null!;
        return base.CompleteChatAsync(request, headers, ct);
    }

    public override IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Tools?.Any() == false) request.Tools = null!;
        return base.StreamChatAsync(request, headers, ct);
    }

    public async Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Stream == true) throw new ProviderRequestException("DigitalOcean does not support streaming image generation.");
        using var client = CreateClient();
        return await CompatibleImageGeneration.GenerateAsync(client, Id, ApiKey(), request, headers, ct);
    }

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct) => throw new ProviderRequestException("DigitalOcean does not support image edits.");
}
