using System.Text.Json;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class ResembleAIProvider
{
    private const string TranscriptionPath = "api/v2/speech-to-text";
    private static readonly HashSet<string> TranscriptionReserved = new(StringComparer.OrdinalIgnoreCase)
        { "model", "file", "stream", "response_format", "providerOptions" };

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!IsTranscriptionModel(model)) throw new ProviderRequestException("ResembleAI supports the speech-to-text transcription model.");
        var native = NativeMediaOptions.Read(Id, options, TranscriptionReserved);
        _ = OptionString(native, "query");
        if (!string.IsNullOrWhiteSpace(language))
        {
            foreach (var name in native.Keys.Where(x => x.Equals("language", StringComparison.OrdinalIgnoreCase)).ToArray()) native.Remove(name);
            native["language"] = JsonSerializer.SerializeToElement(language);
        }

        var started = DateTime.UtcNow;
        using var client = clients.CreateClient(Id);
        var key = RequiredKey(credentials, Id);
        string job;
        using (var form = new MultipartFormDataContent())
        {
            AddFile(form, "file", audio, mediaType, fileName);
            foreach (var (name, value) in native)
                if (value.ValueKind is not (JsonValueKind.Null or JsonValueKind.Undefined))
                    form.Add(new StringContent(value.ValueKind == JsonValueKind.String ? value.GetString()! : value.GetRawText()), name);
            var body = await JsonAsync<JsonElement>(client, HttpMethod.Post, TranscriptionPath, form, "Bearer", key, headers, ct);
            job = String(Property(body, "item"), "uuid") ?? throw new JsonException("ResembleAI did not return a transcription job ID.");
            if (string.IsNullOrWhiteSpace(job) || job is "." or ".." || Property(body, "success").ValueKind == JsonValueKind.False)
                throw new JsonException("ResembleAI returned an invalid transcription job ID.");
        }

        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromMinutes(10));
        var delay = TimeSpan.FromSeconds(1);
        try
        {
            while (true)
            {
                using var response = await MediaHttp.SendAsync(client, HttpMethod.Get, $"{TranscriptionPath}/{Uri.EscapeDataString(job)}",
                    null, "Bearer", key, headers, deadline.Token);
                var body = await NativeHttp.ReadAsync<JsonElement>(response, Json, deadline.Token);
                var item = Property(body, "item");
                var status = String(item, "status")?.ToLowerInvariant();
                if (Property(body, "success").ValueKind == JsonValueKind.False || status == "failed")
                    throw new IOException("ResembleAI transcription failed.");
                if (status == "completed") return Transcript(body, item, model, started, SafeHeaders(response));
                if (status is not ("pending" or "processing")) throw new JsonException("ResembleAI returned an unknown transcription status.");
                await Task.Delay(delay, deadline.Token);
                delay = TimeSpan.FromSeconds(Math.Min(5, delay.TotalSeconds * 1.5));
            }
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested && deadline.IsCancellationRequested)
        {
            throw new IOException("ResembleAI transcription did not complete within ten minutes.");
        }
    }

    private static TranscriptionResponse Transcript(JsonElement body, JsonElement item, string model, DateTime started, Dictionary<string, string> headers)
    {
        var segments = new List<TranscriptionSegment>();
        foreach (var word in Array(Property(item, "words")))
        {
            if (String(word, "text") is not { Length: > 0 } text) continue;
            var start = Seconds(word, "start_time") ?? 0;
            segments.Add(new() { Text = String(word, "speaker_id") is { Length: > 0 } speaker ? $"{speaker}: {text}" : text,
                StartSecond = start, EndSecond = Seconds(word, "end_time") ?? start });
        }
        return new()
        {
            Text = String(item, "text") ?? "", DurationInSeconds = Seconds(item, "duration_seconds"),
            Language = String(item, "language"), Segments = segments, Warnings = [],
            ProviderMetadata = NativeMediaOptions.Metadata(Id, body),
            Response = new() { Timestamp = started, ModelId = model, Body = body, Headers = headers }
        };
    }

    private static float? Seconds(JsonElement node, string name)
        => Property(node, name) is { ValueKind: JsonValueKind.Number } value && value.TryGetSingle(out var seconds) && float.IsFinite(seconds)
            ? seconds : null;
}
