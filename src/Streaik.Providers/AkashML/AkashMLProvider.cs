using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.Messages;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class AkashMLProviderRegistration
{
    public static IServiceCollection AddAkashML(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AkashMLProvider.Id, "https://api.akashml.com/");
        ProviderServiceCollectionExtensions.AddClient(services, AkashMLProvider.AssetClient, "https://api.akashml.com/");
        return services.AddProvider<AkashMLProvider>(AkashMLProvider.Id, "anthropic-version", "anthropic-beta");
    }
}

internal sealed class AkashMLProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id, "v1/"), IMessagesProvider, INativeChatOptionsProvider,
      IImageModelRoutingProvider
{
    internal const string Id = "akashml";
    internal const string AssetClient = "akashml-assets";
    private const string Flux = "black-forest-labs/FLUX.1-schnell";
    private const int ImageLimit = 25 * 1024 * 1024;

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(ConfiguredApiKey())) yield break;
        using var client = CreateClient();
        var result = await NativeHttp.GetAsync<JsonElement>(client, "v1/models", Bearer(ApiKey()), ct);
        var hasFlux = false;
        foreach (var item in Array(Property(result, "data")))
        {
            if (String(item, "id") is not { Length: > 0 } id || Property(item, "active").ValueKind == JsonValueKind.False) continue;
            var isImage = IsFlux(id);
            hasFlux |= isImage;
            yield return new Model
            {
                Id = id, Object = String(item, "object") ?? "model", Name = String(item, "name") ?? id,
                Description = String(item, "description"), OwnedBy = String(item, "owned_by") ?? Id,
                Type = isImage ? "image" : String(item, "type") ?? "language",
                Created = Property(item, "created") is { ValueKind: JsonValueKind.Number } created && created.TryGetInt64(out var time) ? time : null,
                ContextWindow = Integer(item, "context_length"), MaxTokens = Integer(item, "max_output_length"),
                Pricing = Pricing(item)
            };
        }
        if (!hasFlux) yield return new Model { Id = Flux, Name = "FLUX.1 Schnell", OwnedBy = "black-forest-labs", Type = "image" };
    }

    protected override ModelPricing? Pricing(JsonElement model)
        => ProviderPrices.ReadPair(Property(model, "pricing"), "input", "output");

    private static int? Integer(JsonElement item, string field)
        => Property(item, field) is { ValueKind: JsonValueKind.Number } number && number.TryGetInt32(out var value) ? value : null;

    private static bool IsFlux(string model) => model.Equals(Flux, StringComparison.OrdinalIgnoreCase);
    public Task<bool> IsImageModelAsync(string model, CancellationToken ct) => Task.FromResult(IsFlux(model));

    public async Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        return await NativeHttp.PostAsync<MessagesRequest, MessagesResponse>(client, "anthropic/v1/messages", request,
            Bearer(ApiKey()), headers, MessagesJson.Default, ct);
    }

    public async IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        await foreach (var part in NativeHttp.StreamAsync<MessagesRequest, MessageStreamPart>(client, "anthropic/v1/messages", request,
            Bearer(ApiKey()), headers, MessagesJson.Default, part => part.Type is "message_stop" or "error", ct)) yield return part;
    }

    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        ValidateStreamCount(request.Stream, request.N);
        var body = ToObject(request);
        body.Remove("stream");
        body.Remove("partial_images");
        body["response_format"] = "b64_json";
        return ImagesAsync("generations", JsonContent.Create(body, options: Json), headers, ct);
    }

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        ValidateStreamCount(request.Stream, request.N);
        using var form = new MultipartFormDataContent();
        var fields = ToObject(request);
        foreach (var field in fields.Select(p => p.Key).Where(EditReserved.Contains).ToArray()) fields.Remove(field);
        fields["response_format"] = "b64_json";
        AddFields(form, fields);
        foreach (var file in request.ImageFiles ?? []) AddFile(form, "image", file);
        if (request.MaskFile is { } mask) AddFile(form, "mask", mask);
        foreach (var image in request.Images ?? []) form.Add(new StringContent(ImageUrl(image)), "image_url");
        if (request.Mask is { } reference) form.Add(new StringContent(ImageUrl(reference)), "mask_url");
        return await ImagesAsync("edits", form, headers, ct);
    }

    private static readonly HashSet<string> EditReserved = new(StringComparer.OrdinalIgnoreCase)
        { "image", "images", "image_url", "mask", "mask_url", "stream", "partial_images", "response_format" };

    private static void ValidateStreamCount(bool? stream, int? count)
    {
        // AkashML itself does not stream. The gateway wraps one completed image in its existing SSE format.
        if (stream == true && count > 1) throw new ProviderRequestException("Image streaming supports one image; use a non-streaming request for multiple images.");
    }

    private static string ImageUrl(OpenAIImageReference reference)
        => reference.ImageUrl is { Length: > 0 } url ? url : throw new ProviderRequestException("AkashML requires an image URL or uploaded file, not a file ID.");

    private async Task<OpenAIImagesResponse> ImagesAsync(string operation, HttpContent body,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using (body)
        using (var client = CreateClient())
        {
            var result = await JsonAsync<OpenAIImagesResponse>(client, HttpMethod.Post, $"v1/images/{operation}", body, "Bearer", ApiKey(), headers, ct);
            if (result.Data is not { Count: > 0 })
                throw new JsonException("AkashML did not return usable images.");
            foreach (var image in result.Data)
            {
                if (!string.IsNullOrWhiteSpace(image.B64Json)) continue;
#pragma warning disable CS0618 // Normalize the upstream compatibility URL to inline gateway output.
                var (encoded, format) = await DownloadAsync(image.Url, ct);
                image.Url = null;
#pragma warning restore CS0618
                image.B64Json = encoded;
                result.OutputFormat ??= format;
            }
            if (result.AdditionalProperties is { Count: > 0 } extra && !extra.ContainsKey("providerMetadata"))
                extra["providerMetadata"] = JsonSerializer.SerializeToElement(new Dictionary<string, Dictionary<string, JsonElement>> { [Id] = extra }, Json);
            return result;
        }
    }

    private async Task<(string Base64, string Format)> DownloadAsync(string? url, CancellationToken ct)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) || uri.Scheme != Uri.UriSchemeHttps || !uri.IsDefaultPort
            || uri.UserInfo.Length != 0 || uri.Fragment.Length != 0 || uri.Host is not ("api.akashml.com" or "cdn.akashml.com"))
            throw new JsonException("AkashML did not return inline data or a trusted image URL.");
        // Only accept provider-owned delivery hosts. Do not forward credentials or follow redirects.
        using var client = CreateClient(AssetClient);
        using var request = new HttpRequestMessage(HttpMethod.Get, uri);
        using var response = await NativeHttp.SendAsync(client, request, ct);
        var format = response.Content.Headers.ContentType?.MediaType switch
        {
            "image/png" => "png", "image/jpeg" => "jpeg", "image/webp" => "webp",
            _ => throw new JsonException("AkashML asset did not return a supported image type.")
        };
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        return (await BoundedBase64Async(stream, ImageLimit, ct), format);
    }

}
