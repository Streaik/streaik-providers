using System.Globalization;
using System.Text.Json;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class AstriaProvider
{
    public Task<OpenAIImagesResponse> GenerateImageAsync(OpenAIImageGenerationRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => GenerateAsync(request.Model, request.Prompt, request.N, request.Size, request.AdditionalProperties, null, headers, ct);

    public Task<OpenAIImagesResponse> EditImageAsync(OpenAIImageEditRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if ((request.Images?.Length ?? 0) + (request.ImageFiles?.Length ?? 0) != 1)
            throw new ProviderRequestException("Astria accepts one input image; use native reference options for additional references.", 422);
        if (request.Mask is not null && request.MaskFile is not null) throw new ProviderRequestException("Astria accepts one mask.");

        return GenerateAsync(request.Model, request.Prompt, request.N, request.Size, request.AdditionalProperties, (form, options) =>
        {
            if (request.ImageFiles is [var upload]) AddFile(form, "prompt[input_image]", upload);
            else AddReference(form, "input_image", request.Images![0]);
            RemoveMedia(options, "input_image");
            if (request.MaskFile is { } mask) AddFile(form, "prompt[mask_image]", mask);
            else if (request.Mask is { } reference) AddReference(form, "mask_image", reference);
            if (request.MaskFile is not null || request.Mask is not null) RemoveMedia(options, "mask_image");
        }, headers, ct);
    }

    private static void AddReference(MultipartFormDataContent form, string field, OpenAIImageReference reference)
    {
        if (reference.FileId is not null) throw new ProviderRequestException("Astria does not accept OpenAI file IDs.", 422);
        AddMedia(form, field, reference.ImageUrl, null);
    }

    private async Task<OpenAIImagesResponse> GenerateAsync(string model, string prompt, int? n, string? size,
        Dictionary<string, JsonElement>? extras, Action<MultipartFormDataContent, Dictionary<string, JsonElement>>? attach,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(prompt)) throw new ProviderRequestException("An Astria image prompt is required.");
        var tune = await ResolveTuneAsync(model, ct);
        var options = NativeOptions(extras);
        Set(options, "text", prompt);
        Set(options, "num_images", n);
        if (!string.IsNullOrWhiteSpace(size) && size != "auto")
        {
            var parts = size.Split('x', StringSplitOptions.TrimEntries);
            if (parts.Length != 2 || !int.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out var width)
                || !int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out var height) || width <= 0 || height <= 0)
                throw new ProviderRequestException("Astria image size must be WIDTHxHEIGHT.");
            Set(options, "w", width);
            Set(options, "h", height);
        }
        using var form = new MultipartFormDataContent();
        attach?.Invoke(form, options);
        AddOptions(form, options);

        var root = await SubmitAsync(tune, form, headers, ct);
        var id = PromptId(root);
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromMinutes(10));
        try
        {
            while (string.IsNullOrWhiteSpace(String(root, "user_error")) && !Outputs(root).Any())
            {
                root = await PollAsync(tune, id, headers, deadline.Token);
                if (string.IsNullOrWhiteSpace(String(root, "user_error")) && !Outputs(root).Any())
                    await Task.Delay(TimeSpan.FromSeconds(2), deadline.Token);
            }
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested && deadline.IsCancellationRequested)
        { throw new ProviderRequestException("Astria image generation timed out.", 504); }
        if (String(root, "user_error") is { } error && !string.IsNullOrWhiteSpace(error))
            throw new ProviderRequestException(error, 422);

        var images = new List<OpenAIImageData>();
        string? format = null;
        var remaining = MediaLimit;
        foreach (var output in Outputs(root))
        {
            var artifact = (await DownloadAsync(output, false, remaining, ct))!.Value;
            remaining -= artifact.Length;
            format ??= artifact.Mime[6..] switch { "svg+xml" => "svg", var mime => mime };
            images.Add(new() { B64Json = artifact.Data });
        }
        return new() { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Size = size, OutputFormat = format, Data = images,
            AdditionalProperties = new() { ["providerMetadata"] = JsonSerializer.SerializeToElement(Metadata(root), MediaHttp.Json) } };
    }
}
