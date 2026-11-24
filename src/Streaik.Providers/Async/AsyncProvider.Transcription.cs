using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class AsyncProvider
{
    private static readonly HashSet<string> TranscriptionReserved = new(StringComparer.OrdinalIgnoreCase)
        { "file", "model", "model_id", "language", "response_format", "stream" };

    public async Task<TranscriptionResponse> TranscribeAsync(string model, Stream audio, string mediaType, string fileName,
        string? language, IReadOnlyDictionary<string, JsonElement>? options, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!IsTranscriptionModel(model)) throw new ProviderRequestException($"Unsupported asyncAI transcription model '{model}'.");
        using var client = clients.CreateClient(Id);
        using var form = new MultipartFormDataContent();
        AddFile(form, "file", audio, mediaType, fileName);
        form.Add(new StringContent(Asr), "model_id");
        if (!string.IsNullOrWhiteSpace(language)) form.Add(new StringContent(language.Trim()), "language");
        AddFields(form, Options(options, TranscriptionReserved));

        var started = DateTime.UtcNow;
        using var response = await SendAsync(client, "speech_to_text", form, headers, ct);
        var root = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        if (String(root, "text") is not { } text) throw new JsonException("asyncAI did not return transcript text.");
        return new TranscriptionResponse { Text = text, Language = String(root, "language") ?? language,
            DurationInSeconds = Property(root, "duration") is { ValueKind: JsonValueKind.Number } duration && duration.TryGetSingle(out var seconds) ? seconds : null,
            ProviderMetadata = new() { [Id] = root }, Response = new() { ModelId = Asr, Timestamp = started, Body = root } };
    }

    public async Task<HttpResponseMessage> TranscribeNativeAsync(string model, IFormCollection request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var format = request["response_format"].ToString().Trim().ToLowerInvariant();
        if (format is not ("" or "json" or "text" or "verbose_json"))
            throw new ProviderRequestException("Unsupported asyncAI transcription response format.");
        if (request.Files.GetFile("file") is not { Length: > 0 } file) throw new ProviderRequestException("An audio file is required.");
        var options = request.Where(field => !TranscriptionReserved.Contains(field.Key))
            .ToDictionary(field => field.Key, field => JsonSerializer.SerializeToElement(field.Value.ToString()));
        await using var audio = file.OpenReadStream();
        var result = await TranscribeAsync(model, audio, file.ContentType, file.FileName, request["language"], options, headers, ct);

        if (string.Equals(request["stream"], "true", StringComparison.OrdinalIgnoreCase))
        {
            // Async returns a batch transcript. There are no upstream intermediate STT events.
            var delta = string.IsNullOrWhiteSpace(result.Text) ? "" : Event(new { type = "transcript.text.delta", delta = result.Text });
            return new(HttpStatusCode.OK) { Content = new StringContent(delta + Event(new { type = "transcript.text.done", text = result.Text }),
                Encoding.UTF8, "text/event-stream") };
        }

        var body = result.ProviderMetadata![Id];
        if (format == "verbose_json")
        {
            var verbose = Copy(body)!.AsObject();
            verbose["language"] ??= result.Language ?? "";
            verbose["duration"] ??= result.DurationInSeconds ?? 0;
            verbose["segments"] ??= new JsonArray();
            verbose["words"] ??= new JsonArray();
            body = JsonSerializer.SerializeToElement(verbose, Json);
        }
        HttpContent content = format == "text" ? new StringContent(result.Text, Encoding.UTF8, "text/plain")
            : JsonContent.Create(body, options: Json);
        return new(HttpStatusCode.OK) { Content = content };

        static string Event<T>(T value) => $"data: {JsonSerializer.Serialize(value, Json)}\n\n";
    }
}
