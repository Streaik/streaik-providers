using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.Messages;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;

namespace Streaik.Providers;

public static class EchoRegistration
{
    public static IServiceCollection AddEcho(this IServiceCollection services)
        => services.AddProvider<EchoProvider>("echo");
}

public sealed partial class EchoProvider : ILocalFixtureProvider, IModelListingProvider, IChatCompletionsProvider, IResponsesProvider, IMessagesProvider
{
    public const string InputRequiredTool = "ai_input_required";
    private const string Plain = "echo", Reasoning = "echo-reasoning", InputRequired = "echo-input-required";
    private const string CallPrefix = "call_echo_input_";
    private static readonly JsonElement Form = JsonSerializer.SerializeToElement(new
    {
        mode = "form", message = "What should I echo?",
        requestedSchema = new { type = "object", properties = new { message = new { type = "string", title = "Message" } }, required = new[] { "message" } }
    });

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        foreach (var (id, name, type, description) in new[]
        {
            (Plain, "Echo", "language", "Echoes the last user message."),
            (Reasoning, "Echo Reasoning", "language", "Echoes the last user message as both reasoning and answer."),
            (InputRequired, "Echo Input Required", "language", "Requests an MCP form and echoes the submitted result."),
            (ImageModel, "Echo Image Fixture", "image", "Returns a static PNG."),
            (VideoModel, "Echo Video Fixture", "video", "Returns a static WebM clip."),
            (SpeechModel, "Echo Speech Fixture", "speech", "Returns a short audio tone."),
            (TranscriptionModel, "Echo Transcription Fixture", "transcription", "Returns a static transcript."),
            (EmbeddingModel, "Echo Embedding Fixture", "embedding", "Returns deterministic hash-based vectors."),
            (RerankModel, "Echo Rerank Fixture", "reranking", "Ranks literal query matches first."),
            (DecisionModel, "Echo Decision Fixture", "decision", "Returns fixed scores and first choices."),
            (RealtimeModel, "Echo Realtime HTTP Fixture", "realtime", "Returns deliberately non-connectable Realtime and Live session fixtures.")
        })
        {
            ct.ThrowIfCancellationRequested();
            yield return new Model { Id = id, Name = name, Description = description, Type = type, OwnedBy = "echo" };
        }
    }

    private sealed record Turn(string Model, string Text, string? CallId)
    {
        public bool Think => Model == Reasoning;
        public string Id { get; } = Guid.NewGuid().ToString("N");
        public long Now { get; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
    }

    private static Turn Reply(string? model, string? text, string? result, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        return model switch
        {
            Plain or Reasoning => new(model, text ?? "", null),
            InputRequired when result is not null => new(model, FormAnswer(result), null),
            InputRequired => new(model, "", CallPrefix + Guid.NewGuid().ToString("N")),
            _ => throw new ProviderRequestException("Unknown Echo language model.", 404)
        };
    }

    private static bool IsFormCall(string? id, string? name)
        => name == InputRequiredTool && id?.StartsWith(CallPrefix, StringComparison.Ordinal) == true;

    private static string FormAnswer(string result)
    {
        try
        {
            using var doc = JsonDocument.Parse(result);
            var form = doc.RootElement;
            if (form.ValueKind == JsonValueKind.Object && form.TryGetProperty("structuredContent", out var wrapped)) form = wrapped;
            if (form.ValueKind == JsonValueKind.Object && form.TryGetProperty("action", out var action) && action.ValueKind == JsonValueKind.String)
                switch (action.GetString())
                {
                    case "decline": return "decline";
                    case "cancel": return "cancel";
                    case "accept" when form.TryGetProperty("content", out var content) && content.ValueKind == JsonValueKind.Object
                        && content.TryGetProperty("message", out var message) && message.ValueKind == JsonValueKind.String:
                        return content.GetRawText();
                }
        }
        catch (JsonException) { }
        throw new ProviderRequestException("Expected an MCP elicitation result with action accept, decline, or cancel.");
    }

    private static IEnumerable<string> Chunks(string text, CancellationToken ct)
    {
        for (var i = 0; i < text.Length;)
        {
            ct.ThrowIfCancellationRequested();
            var length = Math.Min(32, text.Length - i);
            if (i + length < text.Length && char.IsSurrogatePair(text, i + length - 1)) length++;
            yield return text.Substring(i, length);
            i += length;
        }
    }

    private static Dictionary<string, JsonElement> ReasoningField(string text) => new() { ["reasoning_content"] = JsonSerializer.SerializeToElement(text) };

    private static Turn FromChat(ChatCompletionOptions request, CancellationToken ct)
    {
        var messages = request.Messages.ToList();
        string? output = null;
        if (request.Model == InputRequired && messages.Count >= 2 && messages[^1] is { Role: "tool", ToolCallId: { } id }
            && messages[^2] is { Role: "assistant" } call && call.ToolCalls?.OfType<ChatCompletionMessageToolCall>()
                .Any(x => x.Id == id && IsFormCall(x.Id, x.Function.Name)) == true)
            output = ChatMessageContentExtensions.ToText(messages[^1].Content) ?? "";
        var user = messages.LastOrDefault(m => m.Role == "user");
        return Reply(request.Model, user is null ? null : ChatMessageContentExtensions.ToText(user.Content), output, ct);
    }

    public Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var turn = FromChat(request, ct);
        return Task.FromResult(new ChatCompletion
        {
            Id = "chatcmpl_" + turn.Id, Model = turn.Model, Created = turn.Now, Usage = new(),
            Choices = [new() { FinishReason = turn.CallId is null ? "stop" : "tool_calls", Message = new()
            {
                Content = turn.CallId is null ? turn.Text : null,
                AdditionalProperties = turn.Think ? ReasoningField(turn.Text) : null,
                ToolCalls = turn.CallId is { } id ? [new ChatCompletionMessageToolCall
                    { Id = id, Function = new() { Name = InputRequiredTool, Arguments = Form.GetRawText() } }] : null
            }}]
        });
    }

    public async IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        var turn = FromChat(request, ct);
        ChatCompletionUpdate Chunk(ChatCompletionDelta delta, string? finish = null) => new()
        {
            Id = "chatcmpl_" + turn.Id, Created = turn.Now, Model = turn.Model, Choices = [new() { Delta = delta, FinishReason = finish }]
        };
        yield return Chunk(new() { Role = "assistant" });
        if (turn.CallId is { } id)
            yield return Chunk(new() { ToolCalls = [new() { Index = 0, Id = id, Type = "function",
                Function = new() { Name = InputRequiredTool, Arguments = Form.GetRawText() } }] });
        else
        {
            if (turn.Think)
                foreach (var segment in Chunks(turn.Text, ct)) yield return Chunk(new() { AdditionalProperties = ReasoningField(segment) });
            foreach (var segment in Chunks(turn.Text, ct)) yield return Chunk(new() { Content = segment });
        }
        yield return Chunk(new(), turn.CallId is null ? "stop" : "tool_calls");
        if (request.StreamOptions?.IncludeUsage == true)
            yield return new() { Id = "chatcmpl_" + turn.Id, Model = turn.Model, Created = turn.Now, Usage = new() };
    }

    private static Turn FromResponses(ResponseRequest request, CancellationToken ct)
    {
        var items = request.Input?.Items;
        string? output = null;
        if (request.Model == InputRequired && items is { Count: >= 2 } && items[^1] is ResponseFunctionCallOutputItem result
            && items[^2] is ResponseFunctionCallItem call && call.CallId == result.CallId && IsFormCall(call.CallId, call.Name))
            output = result.Output;
        var content = items?.OfType<ResponseInputMessage>().LastOrDefault(m => m.Role == ResponseRole.User)?.Content;
        return Reply(request.Model, request.Input?.Text ?? content?.Text ?? string.Concat(content?.Parts?.OfType<InputTextPart>().Select(x => x.Text) ?? []), output, ct);
    }

    private static ResponseResult Response(Turn turn, bool complete)
    {
        var response = new ResponseResult { Id = "resp_" + turn.Id, Model = turn.Model, CreatedAt = turn.Now,
            Status = complete ? "completed" : "in_progress" };
        if (!complete) return response;
        response.CompletedAt = turn.Now;
        response.Usage = new ResponseUsage { InputTokens = 0, OutputTokens = 0, TotalTokens = 0 };
        if (turn.CallId is { } id)
            response.Output = [new ResponseFunctionCallItem { Id = "fc_" + turn.Id, CallId = id, Name = InputRequiredTool,
                Arguments = Form.GetRawText(), Status = "completed" }];
        else
        {
            var message = new ResponseInputMessage { Id = "msg_" + turn.Id, Role = ResponseRole.Assistant, Status = "completed",
                Content = new(new ResponseContentPart[] { new OutputTextPart { Text = turn.Text, Annotations = [] } }) };
            response.Output = turn.Think ? [new ResponseReasoningItem { Id = "rs_" + turn.Id, Summary = [new() { Text = turn.Text }] }, message] : [message];
        }
        return response;
    }

    public Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => Task.FromResult(Response(FromResponses(request, ct), true));

    public async IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        var turn = FromResponses(request, ct);
        var seq = 0;
        yield return new ResponseCreated { SequenceNumber = seq++, Response = Response(turn, false) };
        if (turn.CallId is { } id)
        {
            var itemId = "fc_" + turn.Id;
            yield return new ResponseOutputItemAdded { SequenceNumber = seq++, Item = new()
                { Type = "function_call", Id = itemId, CallId = id, Name = InputRequiredTool, Arguments = JsonSerializer.SerializeToElement(""), Status = "in_progress" } };
            yield return new ResponseFunctionCallArgumentsDelta { SequenceNumber = seq++, ItemId = itemId, Delta = Form.GetRawText() };
            yield return new ResponseFunctionCallArgumentsDone { SequenceNumber = seq++, ItemId = itemId, Arguments = Form.GetRawText() };
            yield return new ResponseOutputItemDone { SequenceNumber = seq++, Item = new()
                { Type = "function_call", Id = itemId, CallId = id, Name = InputRequiredTool, Arguments = JsonSerializer.SerializeToElement(Form.GetRawText()), Status = "completed" } };
        }
        else
        {
            var textIndex = turn.Think ? 1 : 0;
            if (turn.Think)
            {
                var idReason = "rs_" + turn.Id;
                yield return new ResponseOutputItemAdded { SequenceNumber = seq++, Item = new() { Type = "reasoning", Id = idReason,
                    AdditionalProperties = new() { ["summary"] = JsonSerializer.SerializeToElement(Array.Empty<object>()) } } };
                yield return new ResponseReasoningSummaryPartAdded { SequenceNumber = seq++, ItemId = idReason,
                    Part = new() { Type = "summary_text", Text = "" } };
                foreach (var segment in Chunks(turn.Text, ct))
                    yield return new ResponseReasoningSummaryTextDelta { SequenceNumber = seq++, ItemId = idReason, Delta = segment };
                yield return new ResponseReasoningSummaryTextDone { SequenceNumber = seq++, ItemId = idReason, Text = turn.Text };
                yield return new ResponseReasoningSummaryPartDone { SequenceNumber = seq++, ItemId = idReason,
                    Part = new() { Type = "summary_text", Text = turn.Text } };
                yield return new ResponseOutputItemDone { SequenceNumber = seq++, Item = new() { Type = "reasoning", Id = idReason,
                    AdditionalProperties = new() { ["summary"] = JsonSerializer.SerializeToElement(new[] { new { type = "summary_text", text = turn.Text } }) } } };
            }
            var idText = "msg_" + turn.Id;
            yield return new ResponseOutputItemAdded { SequenceNumber = seq++, OutputIndex = textIndex, Item = new()
                { Type = "message", Id = idText, Role = "assistant", Status = "in_progress", Content = JsonSerializer.SerializeToElement(Array.Empty<object>()) } };
            yield return new ResponseContentPartAdded { SequenceNumber = seq++, OutputIndex = textIndex, ItemId = idText,
                Part = new() { Type = "output_text", Text = "", Annotations = [] } };
            foreach (var segment in Chunks(turn.Text, ct))
                yield return new ResponseOutputTextDelta { SequenceNumber = seq++, Outputindex = textIndex, ItemId = idText, Delta = segment };
            yield return new ResponseOutputTextDone { SequenceNumber = seq++, Outputindex = textIndex, ItemId = idText, Text = turn.Text };
            var part = new ResponseStreamContentPart { Type = "output_text", Text = turn.Text, Annotations = [] };
            yield return new ResponseContentPartDone { SequenceNumber = seq++, OutputIndex = textIndex, ItemId = idText, Part = part };
            yield return new ResponseOutputItemDone { SequenceNumber = seq++, OutputIndex = textIndex, Item = new()
                { Type = "message", Id = idText, Role = "assistant", Status = "completed", Content = JsonSerializer.SerializeToElement(new[] { part }, ResponseJson.Default) } };
        }
        yield return new ResponseCompleted { SequenceNumber = seq, Response = Response(turn, true) };
    }

    private static string MessageText(MessagesContent? content) => content?.Text ??
        string.Concat(content?.Blocks?.Where(b => b.Type == "text").Select(b => b.Text) ?? []);

    private static Turn FromMessages(MessagesRequest request, CancellationToken ct)
    {
        var messages = request.Messages;
        string? output = null;
        if (request.Model == InputRequired && messages.Count >= 2 && messages[^1] is { Role: "user", Content.Blocks: { Count: 1 } parts }
            && parts[0] is { Type: "tool_result", ToolUseId: { } id } && messages[^2] is { Role: "assistant" } assistant
            && assistant.Content.Blocks?.Any(x => x.Type == "tool_use" && x.Id == id && IsFormCall(x.Id, x.Name)) == true)
            output = MessageText(parts[0].Content);
        return Reply(request.Model, MessageText(messages.LastOrDefault(m => m.Role == "user")?.Content), output, ct);
    }

    private static MessagesResponse Message(Turn turn, bool complete) => new()
    {
        Id = "msg_" + turn.Id, Model = turn.Model, Type = "message", Role = "assistant", Usage = new(),
        StopReason = !complete ? null : turn.CallId is null ? "end_turn" : "tool_use",
        Content = !complete ? [] : turn.CallId is { } id ? [new() { Type = "tool_use", Id = id, Name = InputRequiredTool, Input = Form }]
            : turn.Think ? [new() { Type = "thinking", Thinking = turn.Text, Signature = "echo-fixture" }, new() { Type = "text", Text = turn.Text }]
            : [new() { Type = "text", Text = turn.Text }]
    };

    public Task<MessagesResponse> CreateMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => Task.FromResult(Message(FromMessages(request, ct), true));

    public async IAsyncEnumerable<MessageStreamPart> StreamMessageAsync(MessagesRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        var turn = FromMessages(request, ct);
        var result = Message(turn, true);
        yield return new() { Type = "message_start", Message = Message(turn, false) };
        for (var index = 0; index < result.Content.Count; index++)
        {
            var block = result.Content[index];
            yield return new() { Type = "content_block_start", Index = index, ContentBlock = new() { Type = block.Type,
                Id = block.Id, Name = block.Name, Text = block.Text is null ? null : "", Thinking = block.Thinking is null ? null : "",
                Input = block.Input is null ? null : JsonSerializer.SerializeToElement(new { }) } };
            if (block.Input is { } input)
                yield return new() { Type = "content_block_delta", Index = index, Delta = new() { Type = "input_json_delta", PartialJson = input.GetRawText() } };
            else
                foreach (var segment in Chunks(turn.Text, ct))
                    yield return new() { Type = "content_block_delta", Index = index, Delta = block.Type == "thinking"
                        ? new() { Type = "thinking_delta", Thinking = segment } : new() { Type = "text_delta", Text = segment } };
            if (block.Signature is { } signature)
                yield return new() { Type = "content_block_delta", Index = index, Delta = new() { Type = "signature_delta", Signature = signature } };
            yield return new() { Type = "content_block_stop", Index = index };
        }
        yield return new() { Type = "message_delta", Delta = new() { StopReason = result.StopReason }, Usage = result.Usage };
        yield return new() { Type = "message_stop" };
    }
}
