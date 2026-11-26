using System.Runtime.CompilerServices;
using System.Text.Json;
using Streaik.ChatCompletions.Models;
using Streaik.Responses;
using Streaik.Responses.Output;
using Streaik.Responses.Streaming;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class CambaiProvider : IChatCompletionsProvider, IResponsesProvider, INativeChatOptionsProvider
{
    private static readonly HashSet<string> TranslationControls = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "input", "messages", "instructions", "texts", "source_language", "target_language",
        "stream", "stream_options", "metadata", "store", "tools", "tool_choice", "background", "conversation", "previous_response_id"
    };

    public async Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var result = await TranslateAsync(request.Model, ChatTexts(request), request.AdditionalProperties, ct);
        return new()
        {
            Id = $"chatcmpl-{Guid.NewGuid():N}", Model = request.Model, Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            Choices = [new() { Index = 0, FinishReason = "stop", Message = new() { Role = "assistant", Content = string.Join('\n', result.Texts) } }],
            AdditionalProperties = ResponseExtensions(result.Metadata)
        };
    }

    public async IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var result = await TranslateAsync(request.Model, ChatTexts(request), request.AdditionalProperties, ct);
        var id = $"chatcmpl-{Guid.NewGuid():N}";
        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        yield return Update(new() { Role = "assistant" });
        for (var i = 0; i < result.Texts.Length; i++)
            yield return Update(new() { Content = (i == 0 ? "" : "\n") + result.Texts[i] });
        var done = Update(new(), "stop");
        done.AdditionalProperties = ResponseExtensions(result.Metadata);
        yield return done;

        ChatCompletionUpdate Update(ChatCompletionDelta delta, string? finish = null)
            => new() { Id = id, Model = request.Model, Created = created, Choices = [new() { Index = 0, Delta = delta, FinishReason = finish }] };
    }

    public async Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var model = request.Model ?? throw new ProviderRequestException("A CAMBAI model is required.");
        var result = await TranslateAsync(model, ResponseTexts(request), request.AdditionalProperties, ct);
        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        return Response(model, $"resp_{Guid.NewGuid():N}", created, Message($"msg_{Guid.NewGuid():N}", result.Texts), result.Metadata);
    }

    public async IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var model = request.Model ?? throw new ProviderRequestException("A CAMBAI model is required.");
        var result = await TranslateAsync(model, ResponseTexts(request), request.AdditionalProperties, ct);
        var id = $"resp_{Guid.NewGuid():N}";
        var itemId = $"msg_{Guid.NewGuid():N}";
        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var seq = 0;
        yield return new ResponseCreated { SequenceNumber = seq++, Response = Response(model, id, created) };
        yield return new ResponseOutputItemAdded { SequenceNumber = seq++, OutputIndex = 0, Item = StreamItem(Message(itemId)) };
        for (var i = 0; i < result.Texts.Length; i++)
        {
            var text = result.Texts[i];
            yield return new ResponseContentPartAdded { SequenceNumber = seq++, OutputIndex = 0, ContentIndex = i, ItemId = itemId, Part = StreamText("") };
            yield return new ResponseOutputTextDelta { SequenceNumber = seq++, Outputindex = 0, ContentIndex = i, ItemId = itemId, Delta = text };
            yield return new ResponseOutputTextDone { SequenceNumber = seq++, Outputindex = 0, ContentIndex = i, ItemId = itemId, Text = text };
            yield return new ResponseContentPartDone { SequenceNumber = seq++, OutputIndex = 0, ContentIndex = i, ItemId = itemId, Part = StreamText(text) };
        }
        var completed = Message(itemId, result.Texts);
        yield return new ResponseOutputItemDone { SequenceNumber = seq++, OutputIndex = 0, Item = StreamItem(completed) };
        yield return new ResponseCompleted { SequenceNumber = seq, Response = Response(model, id, created, completed, result.Metadata) };

        static ResponseStreamItem StreamItem(ResponseOutputMessage message)
            => JsonSerializer.SerializeToElement(message, ResponseJson.Default).Deserialize<ResponseStreamItem>(ResponseJson.Default)!;
        static ResponseStreamContentPart StreamText(string text)
            => JsonSerializer.SerializeToElement(new { type = "output_text", text, annotations = System.Array.Empty<object>() }, ResponseJson.Default)
                .Deserialize<ResponseStreamContentPart>(ResponseJson.Default)!;
    }

    private async Task<(string[] Texts, Dictionary<string, JsonElement> Metadata)> TranslateAsync(string model, IEnumerable<string> input,
        Dictionary<string, JsonElement>? extra, CancellationToken ct)
    {
        var (source, target) = LanguageIds(model, "translate", pair: true);
        var texts = input.Where(text => !string.IsNullOrWhiteSpace(text)).ToArray();
        if (texts.Length == 0) throw new ProviderRequestException("CAMBAI translation requires user text.");
        var native = NativeMediaOptions.Read(Id, extra, TranslationControls);
        using var payload = JobPayload(native, ("source_language", source), ("target_language", target!.Value), ("texts", texts));
        var job = await StartJobAsync("translate", payload, ct);
        var body = await JobResultAsync("translation-result", job, ct);
        var translated = Property(body, "texts");
        if (translated.ValueKind != JsonValueKind.Array || translated.GetArrayLength() != texts.Length
            || translated.EnumerateArray().Any(value => value.ValueKind != JsonValueKind.String))
            throw new JsonException("CAMBAI returned an invalid translation result.");
        return (translated.EnumerateArray().Select(value => value.GetString()!).ToArray(), NativeMediaOptions.Metadata(Id, job.Raw(body)));
    }

    private static IEnumerable<string> ChatTexts(ChatCompletionOptions request)
    {
        foreach (var message in request.Messages.Where(message => string.Equals(message.Role, "user", StringComparison.OrdinalIgnoreCase)))
        {
            var content = JsonSerializer.SerializeToElement(message.Content, Json);
            if (content.ValueKind == JsonValueKind.String) yield return content.GetString()!;
            else foreach (var part in Array(content))
                if (String(part, "type") == "text" && String(part, "text") is { } text) yield return text;
        }
    }

    private static IEnumerable<string> ResponseTexts(ResponseRequest request)
    {
        if (request.Input?.Text is { } text) { yield return text; yield break; }
        foreach (var message in (request.Input?.Items ?? []).OfType<ResponseInputMessage>().Where(m => m.Role == ResponseRole.User))
        {
            if (message.Content.Text is { } value) yield return value;
            else foreach (var part in (message.Content.Parts ?? []).OfType<InputTextPart>()) yield return part.Text;
        }
    }

    private static ResponseOutputMessage Message(string id, string[]? texts = null)
        => new() { Id = id, Role = ResponseRole.Assistant, Status = texts is null ? "in_progress" : "completed",
            Content = texts?.Select(text => (ResponseOutputContent)new ResponseOutputText { Text = text, Annotations = [] }).ToList() ?? [] };

    private static ResponseResult Response(string model, string id, long created, ResponseOutputMessage? message = null,
        Dictionary<string, JsonElement>? metadata = null)
        => new() { Model = model, Id = id, CreatedAt = created, Status = message?.Status ?? "in_progress",
            CompletedAt = message is null ? null : DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Output = message is null ? [] : [message],
            AdditionalProperties = metadata is null ? null : ResponseExtensions(metadata) };

    private static Dictionary<string, JsonElement> ResponseExtensions(Dictionary<string, JsonElement> metadata)
        => new() { ["providerMetadata"] = JsonSerializer.SerializeToElement(metadata, Json) };
}
