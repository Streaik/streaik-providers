using System.Globalization;
using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Daglo;

internal sealed partial class DagloProvider
{
    private const string TranscriptsPath = "stt/v1/async/transcripts";

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var key = ApiKey();
        var sttConfig = Option(options, "sttConfig");
        language = string.IsNullOrWhiteSpace(language) ? ReadOption(options, "language") ?? String(sttConfig, "language") : language;
        var timeout = Integer(Option(options, "pollTimeoutSeconds")) ?? Integer(Property(Option(options, "polling"), "timeoutSeconds")) ?? 1800;
        var engine = !string.IsNullOrWhiteSpace(model) ? model : ReadOption(options, "model") ?? String(sttConfig, "model") ?? "general";
        var started = DateTime.UtcNow;
        using var client = CreateClient();
        var rid = await SubmitAsync();

        using var polling = CancellationTokenSource.CreateLinkedTokenSource(ct);
        polling.CancelAfter(TimeSpan.FromSeconds(Math.Clamp(timeout, 30, 14_400)));
        var delay = TimeSpan.FromSeconds(1);
        var path = $"{TranscriptsPath}/{Uri.EscapeDataString(rid)}";
        try
        {
            while (true)
            {
                using (var response = await SendAsync(client, HttpMethod.Get, path, null, "Bearer", key, headers, polling.Token))
                {
                    if (response.StatusCode == HttpStatusCode.NoContent) return ToTranscript(default, language, model, started);
                    var body = await response.Content.ReadAsStringAsync(polling.Token);
                    if (!string.IsNullOrWhiteSpace(body))
                    {
                        using var doc = JsonDocument.Parse(body);
                        var root = doc.RootElement;
                        switch (String(root, "status")?.ToLowerInvariant())
                        {
                            case "transcribed": return ToTranscript(root.Clone(), language, model, started);
                            case "input_error": throw new ProviderRequestException("Daglo could not transcribe the supplied audio.", 422);
                            case "transcript_error" or "file_error": throw new HttpRequestException("Daglo transcription failed.");
                        }
                    }
                }
                await Task.Delay(delay, polling.Token);
                delay = TimeSpan.FromSeconds(Math.Min(6, delay.TotalSeconds * 1.5));
            }
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested && polling.IsCancellationRequested)
        {
            throw new ProviderRequestException("Daglo transcription timed out.", 504);
        }

        async Task<string> SubmitAsync()
        {
            using var form = new MultipartFormDataContent();
            AddFile(form, "file", audio, mediaType, fileName);
            form.Add(JsonContent.Create(new { model = engine, language }, options: Json), "sttConfig");
            var result = await JsonAsync<JsonElement>(client, HttpMethod.Post, TranscriptsPath, form, "Bearer", key, headers, ct);
            var id = String(result, "rid");
            return !string.IsNullOrWhiteSpace(id) && id is not ("." or "..") ? id : throw new JsonException("Daglo did not return a transcription ID.");
        }
    }

    public async Task<HttpResponseMessage> TranscribeNativeAsync(string model, IFormCollection request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var format = request["response_format"].ToString().Trim().ToLowerInvariant();
        if (format is not ("" or "json" or "text" or "verbose_json"))
            throw new ProviderRequestException("Unsupported Daglo transcription response format.");
        if (request.Keys.Any(name => name is "chunking_strategy" or "include" or "include[]" or "known_speaker_names" or "known_speaker_names[]"
            or "known_speaker_references" or "known_speaker_references[]"))
            throw new ProviderRequestException("Daglo does not support chunking, include, or known-speaker options.");
        if (request.Files.GetFile("file") is not { Length: > 0 } file) throw new ProviderRequestException("An audio file is required.");
        var options = request.Where(field => field.Key == "pollTimeoutSeconds").ToDictionary(field => field.Key, field => JsonSerializer.SerializeToElement(field.Value.ToString()));
        await using var audio = file.OpenReadStream();
        var result = await TranscribeAsync(model, audio, file.ContentType, file.FileName, request["language"], options, headers, ct);
        if (string.Equals(request["stream"], "true", StringComparison.OrdinalIgnoreCase))
        {
            // Daglo completes the whole batch before it can emit a delta and a final transcript.
            var delta = string.IsNullOrWhiteSpace(result.Text) ? ""
                : Event(new { type = "transcript.text.delta", delta = result.Text });
            return new(HttpStatusCode.OK) { Content = new StringContent(delta + Event(new { type = "transcript.text.done", text = result.Text }),
                Encoding.UTF8, "text/event-stream") };
        }

        HttpContent content = format switch
        {
            "text" => new StringContent(result.Text, Encoding.UTF8, "text/plain"),
            "verbose_json" => JsonContent.Create(new { text = result.Text, language = result.Language ?? "", duration = result.DurationInSeconds ?? 0,
                segments = result.Segments.Select((segment, i) => new { id = i, start = segment.StartSecond, end = segment.EndSecond, text = segment.Text }) }, options: Json),
            _ => JsonContent.Create(new { text = result.Text }, options: Json)
        };
        return new(HttpStatusCode.OK) { Content = content };

        static string Event<T>(T value) => $"data: {JsonSerializer.Serialize(value, Json)}\n\n";
    }

    private static TranscriptionResponse ToTranscript(JsonElement root, string? language, string model, DateTime started)
    {
        var text = new List<string>();
        var segments = new List<TranscriptionSegment>();
        var duration = 0f;
        foreach (var result in Array(Property(root, "sttResults")))
        {
            if (String(result, "transcript") is { } transcript && !string.IsNullOrWhiteSpace(transcript)) text.Add(transcript);
            foreach (var word in Array(Property(result, "words")))
            {
                var start = Time(Property(word, "startTime"));
                var end = Math.Max(start, Time(Property(word, "endTime")));
                duration = Math.Max(duration, end);
                if (String(word, "word") is { } value && !string.IsNullOrWhiteSpace(value))
                    segments.Add(new() { Text = value, StartSecond = start, EndSecond = end });
            }
        }
        return new() { Text = string.Join(' ', text), Segments = segments, Language = language, DurationInSeconds = duration > 0 ? duration : null,
            ProviderMetadata = root.ValueKind == JsonValueKind.Object ? new() { [Id] = root } : null,
            Response = new() { ModelId = model, Timestamp = started, Body = root.ValueKind == JsonValueKind.Object ? root : null } };
    }

    private static float Time(JsonElement time)
        => (float)((Integer(Property(time, "seconds")) ?? 0) + (Integer(Property(time, "nanos")) ?? 0) / 1_000_000_000d);

    private static JsonElement Option(IReadOnlyDictionary<string, JsonElement>? options, string name)
        => options?.TryGetValue(name, out var value) == true ? value : default;

    private static string? ReadOption(IReadOnlyDictionary<string, JsonElement>? options, string name)
        => Option(options, name) is { ValueKind: JsonValueKind.String } value ? value.GetString() : null;

    private static int? Integer(JsonElement value) => value.ValueKind switch
    {
        JsonValueKind.Number when value.TryGetInt32(out var n) => n,
        JsonValueKind.String when int.TryParse(value.GetString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var n) => n,
        _ => null
    };
}
