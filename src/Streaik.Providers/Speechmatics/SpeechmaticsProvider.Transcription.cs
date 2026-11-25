using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Speechmatics;

internal sealed partial class SpeechmaticsProvider
{
    private const string JobsPath = "v2/jobs";

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (model == AgentModel) throw new ProviderRequestException("Speechmatics Linden 1 supports Agent STT WebSocket transcription only.", 422);
        if (!IsTranscriptionModel(model)) throw new ProviderRequestException("Select a Speechmatics batch transcription model.");

        var transcription = Options(options, "model", "audio", "file", "data_file", "config", "stream", "response_format");
        if (model is "melia-1" or "oak-1") language = "multi";
        else if (string.IsNullOrWhiteSpace(language)) language = ReadString(transcription, "language");
        transcription.Remove("language");
        if (!string.IsNullOrWhiteSpace(language)) transcription["language"] = JsonSerializer.SerializeToElement(language, Json);
        transcription["model"] = JsonSerializer.SerializeToElement(model, Json);
        var config = JsonSerializer.Serialize(new { type = "transcription", transcription_config = transcription }, Json);

        var key = RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        JsonElement submitted;
        using (var form = new MultipartFormDataContent())
        {
            AddFile(form, "data_file", audio, mediaType, fileName);
            form.Add(new StringContent(config), "config");
            submitted = await JsonAsync<JsonElement>(client, HttpMethod.Post, JobsPath, form, "Bearer", key, headers, ct);
        }
        var id = String(submitted, "id") ?? String(Property(submitted, "job"), "id");
        if (string.IsNullOrWhiteSpace(id) || id is "." or "..") throw new JsonException("Speechmatics did not return a transcription job ID.");

        var path = $"{JobsPath}/{Uri.EscapeDataString(id)}";
        using var polling = CancellationTokenSource.CreateLinkedTokenSource(ct);
        polling.CancelAfter(TimeSpan.FromMinutes(6));
        var token = polling.Token;
        try
        {
            while (true)
            {
                var status = await GetAsync(path, token);
                var job = Property(status, "job");
                switch ((String(status, "status") ?? String(job, "status"))?.ToLowerInvariant())
                {
                    case "done":
                        var transcript = await GetAsync($"{path}/transcript?format=json-v2", token);
                        return ParseTranscript(transcript, Number(Property(job, "duration")) ?? Number(Property(status, "duration")), model, language);
                    case "rejected": throw new ProviderRequestException("Speechmatics rejected the transcription job.", 422);
                    case "failed" or "deleted" or "expired" or "cancelled": throw new HttpRequestException("Speechmatics transcription job failed.");
                }
                await Task.Delay(TimeSpan.FromSeconds(2), token);
            }
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested && polling.IsCancellationRequested)
        {
            throw new ProviderRequestException("Speechmatics transcription timed out.", 504);
        }

        Task<JsonElement> GetAsync(string target, CancellationToken cancellation)
            => JsonAsync<JsonElement>(client, HttpMethod.Get, target, null, "Bearer", key, headers, cancellation);
    }

    public async Task<HttpResponseMessage> TranscribeNativeAsync(string model, IFormCollection request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var format = request["response_format"].ToString().Trim().ToLowerInvariant();
        if (format is not ("" or "json" or "text" or "verbose_json"))
            throw new ProviderRequestException("Unsupported Speechmatics transcription response format.");
        if (request.Keys.Any(name => name is "chunking_strategy" or "include" or "include[]" or "known_speaker_names" or "known_speaker_names[]"
            or "known_speaker_references" or "known_speaker_references[]"))
            throw new ProviderRequestException("Speechmatics does not support chunking, include, or known-speaker options.");
        if (request.Files.GetFile("file") is not { Length: > 0 } file) throw new ProviderRequestException("An audio file is required.");
        var options = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (var (name, value) in request)
        {
            if (name is "model" or "file" or "language" or "response_format" or "stream") continue;
            if (name is Id or "providerOptions" or "provider_options")
            {
                using var document = JsonDocument.Parse(value.ToString());
                options[name] = document.RootElement.Clone();
            }
            else if (name is "timestamp_granularities" or "timestamp_granularities[]")
                options["timestamp_granularities"] = JsonSerializer.SerializeToElement(value.ToArray(), Json);
            else if (name == "temperature" && double.TryParse(value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var temperature)
                && double.IsFinite(temperature)) options[name] = JsonSerializer.SerializeToElement(temperature, Json);
            else options[name] = JsonSerializer.SerializeToElement(value.ToString(), Json);
        }

        await using var audio = file.OpenReadStream();
        var result = await TranscribeAsync(model, audio, file.ContentType, file.FileName, request["language"], options, headers, ct);
        HttpContent content;
        if (string.Equals(request["stream"], "true", StringComparison.OrdinalIgnoreCase))
        {
            var delta = string.IsNullOrWhiteSpace(result.Text) ? "" : Event(new { type = "transcript.text.delta", delta = result.Text });
            content = new StringContent(delta + Event(new { type = "transcript.text.done", text = result.Text }), Encoding.UTF8, "text/event-stream");
        }
        else content = format switch
        {
            "text" => new StringContent(result.Text, Encoding.UTF8, "text/plain"),
            "verbose_json" => JsonContent.Create(new { text = result.Text, language = result.Language ?? "", duration = result.DurationInSeconds ?? 0,
                segments = result.Segments.Select((segment, i) => new { id = i, start = segment.StartSecond, end = segment.EndSecond, text = segment.Text }) }, options: Json),
            _ => JsonContent.Create(new { text = result.Text }, options: Json)
        };
        return new(HttpStatusCode.OK) { Content = content };

        static string Event<T>(T value) => $"data: {JsonSerializer.Serialize(value, Json)}\n\n";
    }

    private static TranscriptionResponse ParseTranscript(JsonElement root, float? duration, string model, string? language)
    {
        var text = new StringBuilder();
        var segments = new List<TranscriptionSegment>();
        foreach (var item in Array(Property(root, "results")))
        {
            if (String(Array(Property(item, "alternatives")).FirstOrDefault(), "content") is not { Length: > 0 } word) continue;
            var attaches = String(item, "attaches_to");
            if (text.Length > 0 && !string.Equals(attaches, "previous", StringComparison.OrdinalIgnoreCase)) text.Append(' ');
            text.Append(word);
            if (String(item, "type") is "word" or "punctuation" && Number(Property(item, "start_time")) is { } start
                && Number(Property(item, "end_time")) is { } end)
                segments.Add(new() { Text = word, StartSecond = start, EndSecond = end });
        }
        return new() { Text = text.ToString(), Segments = segments, DurationInSeconds = duration,
            Language = String(Property(Property(root, "metadata"), "transcription_config"), "language") ?? language,
            ProviderMetadata = new() { [Id] = root }, Response = new() { ModelId = model, Timestamp = DateTime.UtcNow } };
    }

    private static float? Number(JsonElement value) => value.ValueKind == JsonValueKind.Number && value.TryGetSingle(out var number)
        && float.IsFinite(number) ? number : null;
}
