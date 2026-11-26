using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.Messages;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class CommonstackProviderRegistration
{
    public static IServiceCollection AddCommonstack(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, CommonstackProvider.Id, "https://api.commonstack.ai/v1/");
        return services.AddProvider<CommonstackProvider>(CommonstackProvider.Id, "anthropic-version", "anthropic-beta");
    }
}

internal sealed class CommonstackProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), IResponsesProvider, IMessagesProvider, INativeChatOptionsProvider,
      IImageProvider, IStreamingImagesProvider
{
    internal const string Id = "commonstack";
    private readonly BearerTextEndpoints text = new(clients, credentials, Id);
    private static readonly HashSet<string> ImageControls = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "prompt", "n", "size", "stream", "partial_images", "image", "image[]", "images", "mask", "mask[]", "files",
        "seed", "aspect_ratio", "aspectRatio"
    };

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        var ids = new HashSet<string>(StringComparer.Ordinal);
        await foreach (var model in base.ListModelsAsync(ct))
            if (ids.Add(model.Id)) yield return model;
        if (string.IsNullOrWhiteSpace(ConfiguredApiKey())) yield break;
        foreach (var (route, description) in new (string, string)[]
        {
            ("auto", "Best quality-per-dollar, adapts with difficulty."),
            ("fast", "Cheapest acceptable model."),
            ("best", "Highest quality, cost nearly ignored.")
        })
        {
            var model = $"uncommon-route/{route}";
            if (ids.Add(model)) yield return new() { Id = model, Name = route, Type = "language", OwnedBy = "UncommonRoute", Description = description };
        }
    }

    public Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.CreateResponseAsync(request, headers, ct);

    public IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.StreamResponseAsync(request, headers, ct);

    public Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.CreateMessageAsync(request, headers, ct);

    public IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => text.StreamMessageAsync(request, headers, ct);

    public async Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var fields = ImageFields(request, request.AdditionalProperties);
        var result = await SendImageAsync("openai/images/generations", JsonContent.Create(fields), fields, headers, ct);
        return WithImageWarnings(result, request.AdditionalProperties, request.Size);
    }

    public async Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var fields = ImageFields(request, request.AdditionalProperties);
        fields.Remove("images");
        fields.Remove("mask");
        using var form = new MultipartFormDataContent();
        AddFields(form, fields);
        foreach (var file in request.ImageFiles ?? []) AddFile(form, "image[]", file);
        foreach (var image in request.Images ?? []) AddDataImage(form, "image[]", image.ImageUrl);
        if (request.MaskFile is { } mask) AddFile(form, "mask", mask);
        else if (request.Mask is { } reference) AddDataImage(form, "mask", reference.ImageUrl);
        var result = await SendImageAsync("openai/images/edits", form, fields, headers, ct);
        return WithImageWarnings(result, request.AdditionalProperties, request.Size);
    }

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageGenerationRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        if (request.N is not (null or 1)) throw new ProviderRequestException("Commonstack image streaming requires one image.");
        request.ResponseFormat = "b64_json";
        yield return NativeMediaOptions.CompletedImage(await GenerateImageAsync(request, headers, ct), edit: false);
    }

    public async IAsyncEnumerable<JsonElement> StreamImagesAsync(OpenAIImageEditRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        if (request.N is not (null or 1)) throw new ProviderRequestException("Commonstack image streaming requires one image.");
        yield return NativeMediaOptions.CompletedImage(await EditImageAsync(request, headers, ct), edit: true);
    }

    private static JsonObject ImageFields<T>(T request, Dictionary<string, JsonElement>? options)
    {
        var payload = ToObject(request);
        foreach (var (name, _) in options ?? []) payload.Remove(name);
        payload.Remove("stream");
        payload.Remove("partial_images");
        foreach (var (name, value) in NativeMediaOptions.Read(Id, options, ImageControls))
            if (!payload.Any(field => field.Key.Equals(name, StringComparison.OrdinalIgnoreCase))) payload[name] = Copy(value);
        return payload;
    }

    private async Task<OpenAIImagesResponse> SendImageAsync(string path, HttpContent content, JsonObject fields,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var ownedContent = content;
        using var client = CreateClient();
        using var response = await MediaHttp.SendAsync(client, HttpMethod.Post, path, content, "Bearer", ApiKey(), headers, ct);
        var body = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        var result = body.Deserialize<OpenAIImagesResponse>(Json) ?? throw new JsonException("Missing Commonstack image response.");
        if (result.Data is not { Count: > 0 } || Array(Property(body, "data")).Any(image =>
            string.IsNullOrWhiteSpace(String(image, "b64_json")) && string.IsNullOrWhiteSpace(String(image, "url"))))
            throw new JsonException("Commonstack returned no usable images.");

        result.OutputFormat ??= Setting("output_format") ?? "png";
        result.Size ??= Setting("size");
        result.Quality ??= Setting("quality");
        result.Background ??= Setting("background");
        if (result.Created == 0) result.Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var raw = ToObject(body);
        foreach (var item in raw["data"]!.AsArray())
        {
            item!.AsObject().Remove("b64_json");
            item.AsObject().Remove("url");
        }
        (result.AdditionalProperties ??= new())["providerMetadata"] = JsonSerializer.SerializeToElement(NativeMediaOptions.Metadata(Id, raw), Json);
        return result;

        string? Setting(string name) => fields[name] is JsonValue value && value.TryGetValue<string>(out var text) ? text : null;
    }

    private static OpenAIImagesResponse WithImageWarnings(OpenAIImagesResponse result, Dictionary<string, JsonElement>? options, string? size)
    {
        var warnings = new List<object>();
        if (options?.GetValueOrDefault("seed").ValueKind == JsonValueKind.Number) warnings.Add(new { type = "unsupported", feature = "seed" });
        if (string.IsNullOrWhiteSpace(size) && options?.GetValueOrDefault("aspect_ratio").ValueKind == JsonValueKind.String)
            warnings.Add(new { type = "unsupported", feature = "aspectRatio" });
        if (warnings.Count > 0) result.AdditionalProperties!["warnings"] = JsonSerializer.SerializeToElement(warnings, Json);
        return result;
    }
}
