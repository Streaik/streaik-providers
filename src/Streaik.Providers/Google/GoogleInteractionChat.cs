using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.WebUtilities;
using Streaik.ChatCompletions.Models;
using Streaik.Interactions.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class GoogleGeminiProvider : IChatBridgeProvider
{
    public IChatCompletionsProvider ChatBridge => new InteractionBridge(this);

    private sealed class InteractionBridge(GoogleGeminiProvider provider) : IChatCompletionsProvider
    {
        public Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
            => provider.CompleteInteractionChatAsync(request, ct);

        public IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
            => provider.StreamInteractionChatAsync(request, includeUsage: true, ct);
    }

    public override Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => GoogleModelIds.IsAgent(request.Model) ? CompleteInteractionChatAsync(request, ct) : base.CompleteChatAsync(request, headers, ct);

    public override IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => GoogleModelIds.IsAgent(request.Model)
            ? StreamInteractionChatAsync(request, request.StreamOptions?.IncludeUsage == true, ct)
            : base.StreamChatAsync(request, headers, ct);

    private async Task<ChatCompletion> CompleteInteractionChatAsync(ChatCompletionOptions request, CancellationToken ct)
    {
        var body = ChatInteraction(request, false);
        var id = default(string);
        try
        {
            using var response = await CreateInteractionAsync(body, ct);
            var result = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
            id = String(result, "id");
            result = await WaitForChatAsync(result, id, body.Store == true, ct);
            var calls = FunctionSteps(result).Select(FunctionCall).ToArray();
            return new ChatCompletion { Id = "chatcmpl-" + Guid.NewGuid().ToString("N"), Model = request.Model,
                Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Usage = ParseInteractionUsage(Property(result, "usage")),
                Choices = [new() { Index = 0, FinishReason = FinishReason(result, calls.Length), Message = new()
                    { Content = ChatText(result), ToolCalls = calls.Length > 0 ? calls : null } }] };
        }
        finally { if (body.Store == true) await CleanUpAgentRunAsync(id); }
    }

    private async IAsyncEnumerable<ChatCompletionUpdate> StreamInteractionChatAsync(ChatCompletionOptions request, bool includeUsage,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var body = ChatInteraction(request, true);
        var completionId = "chatcmpl-" + Guid.NewGuid().ToString("N");
        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var id = default(string);
        var result = default(JsonElement);
        var accumulated = new StringBuilder();
        var streamSteps = new InteractionChatSteps();
        ChatCompletionUsage? usageFromStream = null;
        try
        {
            using (var response = await CreateInteractionAsync(body, ct))
            {
                if (response.Content.Headers.ContentType?.MediaType == "application/json")
                {
                    result = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
                    id = String(result, "id");
                }
                else
                {
                    await foreach (var item in NativeHttp.ReadSseAsync<JsonElement>(response, Json, _ => false, ct))
                    {
                        var type = String(item, "event_type");
                        if (type == "error") throw InteractionFailed();
                        if (Property(item, "interaction") is { ValueKind: JsonValueKind.Object } interaction)
                        {
                            result = interaction;
                            id ??= String(interaction, "id");
                        }
                        // total_usage is cumulative, and can arrive on events without a step index.
                        if (includeUsage)
                            usageFromStream = ParseInteractionUsage(Property(Property(item, "metadata"), "total_usage")) ?? usageFromStream;
                        streamSteps.Observe(item, type);
                        var delta = Property(item, "delta");
                        if (type == "step.delta" && String(delta, "type") == "text" && String(delta, "text") is { Length: > 0 } text)
                        {
                            accumulated.Append(text);
                            yield return Chunk(text);
                        }
                        if (type == "interaction.completed") break;
                    }
                }
            }
            result = await WaitForChatAsync(result, id, body.Store == true, ct);
            if (ChatText(result) is { } full && full.StartsWith(accumulated.ToString(), StringComparison.Ordinal) && full.Length > accumulated.Length)
                yield return Chunk(full[accumulated.Length..]);
            var steps = streamSteps.Complete(result);
            var calls = steps.Where(step => String(step, "type") == "function_call").Select(FunctionCall).ToList();
            for (var i = 0; i < calls.Count; i++)
            {
                var call = calls[i];
                yield return new ChatCompletionUpdate { Id = completionId, Created = created, Model = request.Model,
                    Choices = [new() { Index = 0, Delta = new() { Role = "assistant", ToolCalls = [new()
                        { Index = i, Id = call.Id, Type = "function", Function = new() { Name = call.Function.Name, Arguments = call.Function.Arguments } }] } }] };
            }
            yield return new ChatCompletionUpdate { Id = completionId, Created = created, Model = request.Model,
                Choices = [new() { Index = 0, Delta = new(), FinishReason = FinishReason(result, calls.Count) }],
                AdditionalProperties = body.Store == false ? CaptureInteractionReplay(request.Model, steps) : null };
            if (includeUsage && (ParseInteractionUsage(Property(result, "usage")) ?? usageFromStream) is { } usage)
                yield return new ChatCompletionUpdate { Id = completionId, Created = created, Model = request.Model, Usage = usage };
        }
        finally { if (body.Store == true) await CleanUpAgentRunAsync(id); }

        ChatCompletionUpdate Chunk(string text) => new() { Id = completionId, Created = created, Model = request.Model,
            Choices = [new() { Index = 0, Delta = new() { Role = "assistant", Content = text } }] };
    }

    private static InteractionRequest ChatInteraction(ChatCompletionOptions request, bool stream)
    {
        var isAgent = GoogleModelIds.IsAgent(request.Model);
        var custom = request.Model.StartsWith("agents/", StringComparison.OrdinalIgnoreCase);
        var model = isAgent ? ResourceId(custom ? request.Model[7..] : request.Model) : NormalizeModel(request.Model);
        var native = JsonSerializer.SerializeToElement(request, Json);
        var steps = new JsonArray();
        var functionNames = new Dictionary<string, string>(StringComparer.Ordinal);
        var instructions = new List<string>();
        foreach (var message in Array(Property(native, "messages")))
        {
            var content = Property(message, "content");
            switch (String(message, "role"))
            {
                case "system": case "developer": instructions.Add(ReadText(content)); break;
                case "user": AddStep("user_input", ChatContent(content)); break;
                case "assistant":
                    var output = ChatContent(content);
                    var calls = new List<JsonObject>();
                    foreach (var call in Array(Property(message, "tool_calls")))
                    {
                        var function = Property(call, "function");
                        if (String(call, "type") != "function" || String(call, "id") is not { } encoded
                            || String(function, "name") is not { } name) throw UnsupportedTool();
                        var (id, signature) = DecodeCall(encoded);
                        functionNames[encoded] = name;
                        var next = new JsonObject { ["type"] = "function_call", ["id"] = id, ["name"] = name,
                            ["arguments"] = ParseArguments(String(function, "arguments")) };
                        if (signature is not null) next["signature"] = signature;
                        calls.Add(next);
                    }
                    if (!isAgent && TryReplayInteractionRound(Property(message, InteractionReplayKey), request.Model, output, calls) is { } replay)
                    {
                        foreach (var step in replay) steps.Add(step);
                    }
                    else
                    {
                        AddStep("model_output", output);
                        foreach (var call in calls) steps.Add(call);
                    }
                    break;
                case "tool":
                    if (String(message, "tool_call_id") is not { } resultId || !functionNames.TryGetValue(resultId, out var toolName))
                        throw new ProviderRequestException("Google requires the original function call before its result.", 422);
                    steps.Add(new JsonObject { ["type"] = "function_result", ["call_id"] = DecodeCall(resultId).Id,
                        ["name"] = toolName, ["result"] = ChatContent(content) });
                    break;
                default: throw new ProviderRequestException("Unsupported Google chat message role.", 422);
            }
        }
        if (steps.Count == 0) throw new ProviderRequestException("Google chat requires input.");
        var body = new InteractionRequest { Agent = isAgent ? model : null, Model = isAgent ? null : model, Store = isAgent, Stream = stream,
            Background = isAgent && !custom && model.StartsWith("deep-research", StringComparison.OrdinalIgnoreCase) ? true : null,
            SystemInstruction = instructions.Count > 0 ? string.Join("\n\n", instructions) : null,
            Input = new InteractionsInput(JsonSerializer.SerializeToElement(steps, Json)) };

        var tools = new JsonArray();
        foreach (var tool in Array(Property(native, "tools")))
        {
            if (String(tool, "type") != "function" || Property(tool, "function") is not { ValueKind: JsonValueKind.Object } function) throw UnsupportedTool();
            var mapped = (JsonObject)Copy(function)!;
            mapped.Remove("strict");
            mapped["type"] = "function";
            tools.Add(mapped);
        }
        if (tools.Count > 0)
            body.Tools = tools.Deserialize<List<InteractionTool>>(InteractionJson.Default);

        if (!isAgent)
        {
            var generation = new JsonObject();
            foreach (var field in new[] { "temperature", "top_p", "stop" })
                if (Property(native, field) is { ValueKind: not (JsonValueKind.Null or JsonValueKind.Undefined) } value)
                    generation[field == "stop" ? "stop_sequences" : field] = field == "stop" && value.ValueKind == JsonValueKind.String
                        ? new JsonArray(JsonValue.Create(value.GetString())) : Copy(value);
            var max = Property(native, "max_completion_tokens");
            if (max.ValueKind != JsonValueKind.Number) max = Property(native, "max_tokens");
            if (max.ValueKind == JsonValueKind.Number) generation["max_output_tokens"] = Copy(max);
            var choice = Property(native, "tool_choice");
            if (choice.ValueKind == JsonValueKind.String)
                generation["tool_choice"] = choice.GetString() switch { "required" => "any", "none" => "none", _ => "auto" };
            else if (String(Property(choice, "function"), "name") is { Length: > 0 } only)
                generation["tool_choice"] = new JsonObject { ["mode"] = "any", ["allowed_function_names"] = new JsonArray(JsonValue.Create(only)) };
            if (generation.Count > 0) body.GenerationConfig = generation.Deserialize<InteractionGenerationConfig>(InteractionJson.Default);
            var responseFormat = Property(native, "response_format");
            if (String(responseFormat, "type") is "json_schema" or "json_object")
            {
                var format = new JsonObject { ["type"] = "text", ["mime_type"] = "application/json" };
                var schema = Property(Property(responseFormat, "json_schema"), "schema");
                if (schema.ValueKind == JsonValueKind.Object) format["schema"] = Copy(schema);
                body.ResponseFormat = format.Deserialize<InteractionResponseFormats>(InteractionJson.Default);
            }
        }
        return body;

        void AddStep(string type, JsonArray content)
        {
            if (content.Count > 0) steps.Add(new JsonObject { ["type"] = type, ["content"] = content });
        }
    }

    private static JsonArray ChatContent(JsonElement value)
    {
        if (value.ValueKind == JsonValueKind.String) return new JsonArray(TextPart(value.GetString()!));
        var output = new JsonArray();
        foreach (var part in Array(value))
        {
            switch (String(part, "type"))
            {
                case "text": output.Add(TextPart(String(part, "text") ?? "")); break;
                case "image_url": output.Add(GoogleImagePart(String(Property(part, "image_url"), "url"), null)); break;
                case "input_audio":
                    var audio = Property(part, "input_audio");
                    var format = String(audio, "format");
                    output.Add(new JsonObject { ["type"] = "audio", ["data"] = String(audio, "data"),
                        ["mime_type"] = format switch { "mp3" => "audio/mpeg", "wav" => "audio/wav", _ => throw new ProviderRequestException("Use WAV or MP3 audio.", 422) } });
                    break;
                case "file":
                    var file = Property(part, "file");
                    if (String(file, "file_data") is not { } data || !data.StartsWith("data:application/pdf;base64,", StringComparison.Ordinal))
                        throw new ProviderRequestException("Google compatible file inputs require inline PDF data.", 422);
                    output.Add(new JsonObject { ["type"] = "document", ["mime_type"] = "application/pdf", ["data"] = data[28..] });
                    break;
                default: throw new ProviderRequestException("Unsupported Google chat content part.", 422);
            }
        }
        return output;
    }

    private static string ReadText(JsonElement content) => content.ValueKind == JsonValueKind.String ? content.GetString()!
        : string.Concat(Array(content).Select(part => String(part, "text") ?? throw new ProviderRequestException("Google instructions require text.", 422)));

    private static JsonObject ParseArguments(string? value)
    {
        try { return string.IsNullOrWhiteSpace(value) ? new JsonObject() : JsonNode.Parse(value)?.AsObject() ?? throw UnsupportedTool(); }
        catch (Exception ex) when (ex is JsonException or InvalidOperationException) { throw UnsupportedTool(); }
    }

    private static ChatCompletionMessageToolCall FunctionCall(JsonElement step)
    {
        var id = String(step, "id") ?? throw new JsonException("Missing Google function call ID.");
        var name = String(step, "name") ?? throw new JsonException("Missing Google function name.");
        var args = Property(step, "arguments");
        return new() { Id = EncodeCall(id, String(step, "signature")), Function = new() { Name = name,
            Arguments = args.ValueKind switch { JsonValueKind.Object => args.GetRawText(), JsonValueKind.String => args.GetString()!, _ => "{}" } } };
    }

    // Existing client tool IDs are opaque. Include the signature there so stateless tool replay survives Chat, Responses, and Messages adapters.
    private static string EncodeCall(string id, string? signature) => signature is null ? id
        : "gcall_" + WebEncoders.Base64UrlEncode(JsonSerializer.SerializeToUtf8Bytes(new[] { id, signature }));
    private static (string Id, string? Signature) DecodeCall(string id)
    {
        if (!id.StartsWith("gcall_", StringComparison.Ordinal)) return (id, null);
        try
        {
            if (id.Length <= 65536 && JsonSerializer.Deserialize<string[]>(WebEncoders.Base64UrlDecode(id[6..])) is [var call, var signature])
                return (call, signature);
        }
        catch (Exception ex) when (ex is FormatException or JsonException) { }
        throw UnsupportedTool();
    }

    private static ProviderRequestException UnsupportedTool() => new("Google compatible chat requires standard function tools with JSON object arguments.", 422);
    private static IEnumerable<JsonElement> FunctionSteps(JsonElement result) => Array(Property(result, "steps")).Where(x => String(x, "type") == "function_call");
    private static string? ChatText(JsonElement result) => string.Concat(OutputParts(result).Where(x => String(x, "type") == "text").Select(x => String(x, "text")))
        is { Length: > 0 } text ? text : null;
    private static string FinishReason(JsonElement result, int calls) => calls > 0 ? "tool_calls"
        : String(result, "status") is "incomplete" or "budget_exceeded" ? "length" : "stop";

    private async Task<JsonElement> WaitForChatAsync(JsonElement interaction, string? id, bool stored, CancellationToken ct)
    {
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromMinutes(10));
        try
        {
            while (String(interaction, "status") is not ("completed" or "requires_action" or "incomplete" or "budget_exceeded"))
            {
                if (String(interaction, "status") is not ("pending" or "queued" or "in_progress" or null) || !stored || id is null) throw InteractionFailed();
                await Task.Delay(TimeSpan.FromSeconds(2), deadline.Token);
                using var response = await GetInteractionAsync(id, deadline.Token);
                interaction = await NativeHttp.ReadAsync<JsonElement>(response, Json, deadline.Token);
            }
            return interaction;
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested && deadline.IsCancellationRequested)
        { throw new ProviderRequestException("The Google agent run timed out.", 504); }
    }

    private async Task CleanUpAgentRunAsync(string? id)
    {
        if (id is null) return;
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        try { using var deleted = await DeleteInteractionAsync(id, deadline.Token); }
        catch (Exception ex) when (ex is HttpRequestException or OperationCanceledException or ProviderRequestException or IOException) { }
    }

    private static ProviderRequestException InteractionFailed() => new("The Google request did not complete successfully.", 502);

    private static ChatCompletionUsage? ParseInteractionUsage(JsonElement usage)
    {
        if (Tokens("total_input_tokens") is not { } input || Tokens("total_output_tokens") is not { } output) return null;
        var cached = Tokens("total_cached_tokens");
        var thoughts = Tokens("total_thought_tokens");
        var toolInput = Tokens("total_tool_use_tokens") ?? 0;
        var reportedTotal = Tokens("total_tokens");
        if (input < 0 || output < 0 || cached < 0 || cached > input || thoughts < 0 || toolInput < 0 || reportedTotal < 0) return null;

        // Interactions reports tool prompts and thoughts separately. In the chat contracts,
        // input includes tool prompts and cached input, and output includes reasoning.
        var prompt = (long)input + toolInput;
        var completion = (long)output + (thoughts ?? 0);
        var total = prompt + completion;
        if (total > int.MaxValue) return null;
        return new() { PromptTokens = (int)prompt, CompletionTokens = (int)completion, TotalTokens = reportedTotal ?? (int)total,
            PromptTokensDetails = cached.HasValue ? new() { CachedTokens = cached } : null,
            CompletionTokensDetails = thoughts.HasValue ? new() { ReasoningTokens = thoughts } : null };

        int? Tokens(string key) => Property(usage, key) is { ValueKind: JsonValueKind.Number } value && value.TryGetInt32(out var count) ? count : null;
    }
}
