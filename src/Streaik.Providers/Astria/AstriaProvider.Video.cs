using System.Text.Json;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class AstriaProvider
{
    public async Task<VideoOperationStartResult> StartVideoAsync(VideoRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Prompt)) throw new ProviderRequestException("An Astria video prompt is required.");
        var catalog = await CatalogAsync(ct);
        if (Property(Property(catalog, "video_model_families"), request.Model).ValueKind != JsonValueKind.Object)
            throw new ProviderRequestException("Unknown Astria video model.");
        var defaultTune = String(catalog, "default") ?? throw new JsonException("Astria catalog is missing the default image model.");
        var tune = await ResolveTuneAsync(defaultTune, ct);
        var options = NativeOptions(request.ProviderOptions, scopedOnly: true);
        Set(options, "video_model", request.Model);
        Set(options, "video_prompt", request.Prompt);
        Set(options, "video_resolution", request.Resolution);
        Set(options, "video_duration", request.Duration);
        Set(options, "video_audio", request.GenerateAudio);
        Set(options, "seed", request.Seed);
        Set(options, "num_images", request.N);
        Set(options, "aspect_ratio", request.AspectRatio);

        var first = request.Image;
        VideoFile? last = null;
        foreach (var frame in request.FrameImages ?? [])
            switch (frame.FrameType)
            {
                case "first_frame" when first is null: first = frame.Image; break;
                case "last_frame" when last is null: last = frame.Image; break;
                default: throw new ProviderRequestException("Astria accepts one first frame and one last frame.", 422);
            }
        var inputs = request.InputReferences?.ToArray() ?? [];
        var videos = inputs.Where(input => IsType(input, "video/")).ToArray();
        var audio = inputs.Where(input => IsType(input, "audio/")).ToArray();
        var images = inputs.Where(input => !IsType(input, "video/") && !IsType(input, "audio/")).ToArray();
        if (videos.Length > 1 || audio.Length > 1) throw new ProviderRequestException("Astria accepts one video and one audio reference.", 422);
        if (images.Any(file => IsRemote(Data(file))) && images.Any(file => !IsRemote(Data(file))))
            throw new ProviderRequestException("Astria image references must be all URLs or all binary files to preserve their order.", 422);
        using var form = new MultipartFormDataContent();
        Attach(first, "video_first_frame");
        Attach(last, "video_last_frame");
        Attach(videos.SingleOrDefault(), "input_video");
        Attach(audio.SingleOrDefault(), "audio_reference");
        if (images.Length > 0)
        {
            options.Remove("image_references");
            options.Remove("image_reference_urls");
            foreach (var image in images)
                if (IsRemote(Data(image))) form.Add(new StringContent(Data(image)!), "prompt[image_reference_urls][]");
                else AddBinary(form, "prompt[image_references][]", Data(image), image.MediaType);
        }
        AddOptions(form, options);
        var root = await SubmitAsync(tune, form, headers, ct);
        return new() { Operation = $"{tune}:{PromptId(root)}", ProviderMetadata = Metadata(root),
            Warnings = request.Fps.HasValue ? [new { type = "unsupported", feature = "fps" }] : [],
            Response = new() { ModelId = request.Model, Timestamp = DateTime.UtcNow } };

        void Attach(VideoFile? file, string field)
        {
            if (file is null) return;
            RemoveMedia(options, field);
            AddMedia(form, field, Data(file), file.MediaType);
        }
    }

    public async Task<VideoOperationStatusResult> GetVideoAsync(string operation, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (operation.Split(':') is not [var tuneValue, var promptValue] || PositiveId(tuneValue) is not { } tune
            || PositiveId(promptValue) is not { } prompt) throw new ProviderRequestException("Invalid Astria operation.");
        var root = await PollAsync(tune, prompt, headers, ct);
        if (root.ValueKind != JsonValueKind.Object) throw new JsonException("Invalid Astria prompt status.");
        var response = new HeaderResponseData { ModelId = String(root, "video_model_family") ?? String(root, "video_model") ?? tune,
            Timestamp = DateTime.UtcNow };
        var metadata = Metadata(root);
        if (String(root, "user_error") is { } error && !string.IsNullOrWhiteSpace(error))
            return new VideoOperationErrorResult { Error = error, ProviderMetadata = metadata, Response = response };
        var videos = new List<VideoOperationVideoData>();
        var remaining = MediaLimit;
        foreach (var url in Outputs(root))
            if (await DownloadAsync(url, true, remaining, ct) is { } asset)
            {
                remaining -= asset.Length;
                videos.Add(new() { Type = "base64", Data = asset.Data, MediaType = asset.Mime });
            }
        return videos.Count == 0 ? new VideoOperationPendingResult { ProviderMetadata = metadata, Response = response }
            : new VideoOperationCompletedResult { Videos = videos, ProviderMetadata = metadata, Response = response };
    }

    private static string? Data(VideoFile file) => file is VideoFileUrl url ? url.Url : file.Data;
    private static bool IsType(VideoFile file, string type)
    {
        if (file.MediaType is { Length: > 0 } mime) return mime.StartsWith(type, StringComparison.OrdinalIgnoreCase);
        // The existing URL variant does not carry a media type; infer this from its path, not from a network probe.
        if (!Uri.TryCreate(Data(file), UriKind.Absolute, out var url) || !IsRemote(Data(file))) return false;
        var extension = Path.GetExtension(url.AbsolutePath).ToLowerInvariant();
        return type == "video/" ? extension is ".mp4" or ".webm" or ".mov" or ".m4v" or ".avi"
            : type == "audio/" && extension is ".mp3" or ".wav" or ".m4a" or ".ogg" or ".opus" or ".flac" or ".aac";
    }
}
