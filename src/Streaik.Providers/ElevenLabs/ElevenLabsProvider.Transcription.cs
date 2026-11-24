using System.Globalization;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class ElevenLabsProvider
{
    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!IsTranscriptionModel(model)) throw new ProviderRequestException("Unsupported ElevenLabs file transcription model.");
        var fields = Options(options);
        var logging = TakeQuery(fields, "enable_logging");
        language = string.IsNullOrWhiteSpace(language) ? Text(fields["language_code"]) ?? Text(fields["language"]) : language;
        if (fields["timestamps_granularity"] is null && fields["timestamp_granularities"] is { } granularities
            && (granularities is JsonArray levels ? levels.Any(level => Text(level) == "word") : Text(granularities) == "word"))
            fields["timestamps_granularity"] = "word";
        foreach (var reserved in new[] { "model", "model_id", "file", "language", "language_code", "stream", "response_format",
            "timestamp_granularities", "timestamp_granularities[]", "cloud_storage_url" }) fields.Remove(reserved);
        if (Flag(fields["webhook"])) throw new ProviderRequestException("This transcription route requires a synchronous ElevenLabs response.");
        fields["model_id"] = model;
        if (!string.IsNullOrWhiteSpace(language)) fields["language_code"] = language;

        using var form = new MultipartFormDataContent();
        AddFile(form, "file", audio, mediaType, fileName);
        AddFields(form, fields);
        using var client = clients.CreateClient(Id);
        using var upstream = await SendAsync(client, HttpMethod.Post, "speech-to-text" + (logging is null ? "" : "?" + logging), form, headers, ct);
        var body = await NativeHttp.ReadAsync<JsonElement>(upstream, Json, ct);
        var segments = Array(Property(body, "words")).Where(word => !string.IsNullOrWhiteSpace(String(word, "text")))
            .Select(word => new TranscriptionSegment { Text = String(word, "text")!, StartSecond = Number(word, "start") ?? 0,
                EndSecond = Number(word, "end") ?? Number(word, "start") ?? 0 }).ToArray();
        var text = String(body, "text") ?? throw new JsonException("Missing ElevenLabs transcription text.");
        return new() { Text = text, Language = String(body, "language_code"), Segments = segments,
            DurationInSeconds = Number(body, "duration") ?? Number(body, "duration_seconds"),
            Response = new() { ModelId = model, Timestamp = DateTime.UtcNow }, ProviderMetadata = Metadata(body) };
    }

    public async Task<HttpResponseMessage> TranscribeNativeAsync(string model, IFormCollection request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var format = request["response_format"].ToString();
        if (format is not ("" or "json" or "text" or "verbose_json" or "srt" or "vtt")) throw new ProviderRequestException("Unsupported transcription response format.");
        var stream = request["stream"].ToString();
        if (stream is not ("" or "false" or "true")) throw new ProviderRequestException("Invalid transcription stream flag.");
        var file = request.Files.GetFile("file") ?? throw new ProviderRequestException("An audio file is required.");
        var fields = request.Where(field => field.Key is not ("model" or "file" or "stream" or "response_format" or "language"))
            .ToDictionary(field => field.Key, field => field.Value.Count > 1 ? JsonSerializer.SerializeToElement(field.Value.ToArray())
                : JsonSerializer.SerializeToElement(field.Value.ToString()), StringComparer.OrdinalIgnoreCase);
        if (fields.Remove("timestamp_granularities[]", out var granularities)) fields["timestamp_granularities"] = granularities;
        await using var audio = file.OpenReadStream();
        var result = await TranscribeAsync(model, audio, file.ContentType, file.FileName, request["language"], fields, headers, ct);
        if (stream == "true")
        {
            var delta = JsonSerializer.Serialize(new { type = "transcript.text.delta", delta = result.Text }, Json);
            var done = JsonSerializer.Serialize(new { type = "transcript.text.done", text = result.Text }, Json);
            return Reply((result.Text.Length == 0 ? "" : $"data: {delta}\n\n") + $"data: {done}\n\n", "text/event-stream");
        }
        return format switch
        {
            "text" => Reply(result.Text, "text/plain"),
            "verbose_json" => Reply(JsonSerializer.Serialize(new { text = result.Text, language = result.Language, duration = result.DurationInSeconds,
                segments = result.Segments.Select((item, index) => new { id = index, start = item.StartSecond, end = item.EndSecond, text = item.Text }) }, Json), "application/json"),
            "srt" or "vtt" => Reply(Captions(result, format == "vtt"), format == "vtt" ? "text/vtt" : "text/plain"),
            _ => Reply(JsonSerializer.Serialize(new { text = result.Text }, Json), "application/json")
        };
    }

    private static string Captions(TranscriptionResponse result, bool vtt)
    {
        var lines = result.Segments.ToArray();
        if (lines.Length == 0 && result.Text.Length > 0) lines = [new() { Text = result.Text, EndSecond = result.DurationInSeconds ?? 0 }];
        string Time(double seconds) => TimeSpan.FromSeconds(Math.Clamp(seconds, 0, 359999))
            .ToString(vtt ? @"hh\:mm\:ss\.fff" : @"hh\:mm\:ss\,fff", CultureInfo.InvariantCulture);
        return (vtt ? "WEBVTT\n\n" : "") + string.Join('\n', lines.Select((line, i) =>
            $"{i + 1}\n{Time(line.StartSecond)} --> {Time(line.EndSecond)}\n{line.Text}\n"));
    }

    private static HttpResponseMessage Reply(string body, string type) => new(HttpStatusCode.OK) { Content = new StringContent(body, Encoding.UTF8, type) };
    private static double? Number(JsonElement body, string name) => Property(body, name) is { ValueKind: JsonValueKind.Number } value
        && value.TryGetDouble(out var number) && double.IsFinite(number) ? number : null;
}
