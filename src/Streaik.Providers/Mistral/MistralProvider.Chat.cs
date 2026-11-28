using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.ChatCompletions.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Mistral;

internal sealed partial class MistralProvider
{
    public override async Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var body = ChatBody(request, false);
        if (IsOcrModel(request.Model))
        {
            var result = await ReadOcrAsync(request.Model, ChatOcrFiles(body), body, headers, ct);
            return JsonSerializer.SerializeToElement(new { id = result.Id, @object = "chat.completion", created = result.Created,
                model = request.Model, choices = new[] { new { index = 0, message = new { role = "assistant", content = result.Text }, finish_reason = "stop" } },
                mistral = new { pages_processed = result.Pages } }).Deserialize<ChatCompletion>(Json)!;
        }
        using var client = CreateClient();
        var raw = await NativeHttp.PostAsync<JsonObject, JsonElement>(client, "chat/completions", body, Bearer(ApiKey()), headers, Json, ct);
        return ChatFrame<ChatCompletion>(raw, "message");
    }

    public override async IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        var body = ChatBody(request, true);
        if (IsOcrModel(request.Model))
        {
            var result = await ReadOcrAsync(request.Model, ChatOcrFiles(body), body, headers, ct);
            yield return Chunk(new { role = "assistant", content = result.Text }, null);
            yield return Chunk(new { }, "stop");
            yield break;

            ChatCompletionUpdate Chunk(object delta, string? finish) => JsonSerializer.SerializeToElement(new { id = result.Id,
                @object = "chat.completion.chunk", created = result.Created, model = request.Model,
                choices = new[] { new { index = 0, delta, finish_reason = finish } }, mistral = new { pages_processed = result.Pages } })
                .Deserialize<ChatCompletionUpdate>(Json)!;
        }

        using var client = CreateClient();
        await foreach (var part in NativeHttp.StreamAsync<JsonObject, JsonElement>(client, "chat/completions", body,
            Bearer(ApiKey()), headers, Json, _ => false, ct))
            yield return ChatFrame<ChatCompletionUpdate>(Unwrap(part, "choices"), "delta");
    }

    private static JsonObject ChatBody(ChatCompletionOptions request, bool stream)
    {
        var body = JsonSerializer.SerializeToNode(NativeOptionsPayload.Build(request, Id, Json, stream))!.AsObject();
        foreach (var unsupported in new[] { "store", "stream_options", "logit_bias", "modalities", "audio", "verbosity", "user",
            "web_search_options", "safety_identifier", "prompt_cache_retention", "prompt_cache_options", "moderation", "functions", "function_call" })
            body.Remove(unsupported);
        Rename(body, "max_completion_tokens", "max_tokens", overwrite: true);
        Rename(body, "seed", "random_seed");
        if (body["response_format"] is JsonObject format && format["json_schema"] is JsonObject schema)
            Rename(schema, "schema", "schema_definition");

        if (body["messages"] is JsonArray messages)
        {
            var toolNames = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var message in messages.OfType<JsonObject>())
            {
                var role = NodeString(message["role"]);
                if (role == "developer") message["role"] = "system";
                if (message["tool_calls"] is JsonArray calls)
                    foreach (var call in calls.OfType<JsonObject>())
                        if (NodeString(call["id"]) is { } id && call["function"] is JsonObject function
                            && NodeString(function["name"]) is { } name)
                            toolNames[id] = name;
                if (role == "tool" && NodeString(message["tool_call_id"]) is { } toolId
                    && message["name"] is null && toolNames.TryGetValue(toolId, out var toolName))
                    message["name"] = toolName;

                if (message["content"] is not JsonArray content) continue;
                for (var index = 0; index < content.Count; index++)
                {
                    if (content[index] is not JsonObject part) continue;
                    if (NodeString(part["type"]) == "file" && part["file"] is JsonObject file)
                    {
                        content[index] = NodeString(file["file_id"]) is { Length: > 0 } fileId
                            ? new JsonObject { ["type"] = "file", ["file_id"] = fileId }
                            : new JsonObject { ["type"] = "document_url", ["document_url"] = file["file_data"]?.DeepClone(),
                                ["document_name"] = file["filename"]?.DeepClone() };
                    }
                    else if (NodeString(part["type"]) == "input_audio" && part["input_audio"] is JsonObject audio)
                        part["input_audio"] = audio["data"]?.DeepClone();
                    else if (role == "assistant" && NodeString(part["type"]) == "thinking" && NodeString(part["thinking"]) is { } thought)
                        // The gateway replays thinking as text; Mistral accepts typed chunks inside a thinking block.
                        part["thinking"] = new JsonArray(new JsonObject { ["type"] = "text", ["text"] = thought });
                }
            }
        }
        return body;
    }

    private static T ChatFrame<T>(JsonElement raw, string messageField)
    {
        if (Property(raw, "choices").ValueKind != JsonValueKind.Array)
            throw new JsonException("Mistral Chat Completions event did not contain choices.");
        var body = JsonNode.Parse(raw.GetRawText())!.AsObject();
        foreach (var choice in body["choices"]!.AsArray().OfType<JsonObject>())
        {
            if (choice[messageField] is not JsonObject message) continue;
            if (message["content"] is JsonArray content)
            {
                var text = TextChunks(content);
                var reasoning = string.Concat(content.OfType<JsonObject>().Where(part => NodeString(part["type"]) == "thinking")
                    .Select(part => part["thinking"] is JsonArray chunks ? TextChunks(chunks) : NodeString(part["thinking"])));
                message["content"] = text.Length > 0 ? JsonValue.Create(text) : null;
                if (reasoning.Length > 0) message["reasoning_content"] = reasoning;
                // Preserve native annotations and any other output chunks not representable as plain text.
                message["mistral_content"] = content;
            }
            if (message["tool_calls"] is JsonArray calls)
                foreach (var call in calls.OfType<JsonObject>())
                    if (call["function"] is JsonObject function && function["arguments"] is JsonNode arguments
                        && NodeString(arguments) is null)
                        function["arguments"] = arguments.ToJsonString();
        }
        return body.Deserialize<T>(Json) ?? throw new JsonException("Missing Mistral Chat Completions response.");
    }

    private static string TextChunks(JsonArray content) => string.Concat(content.OfType<JsonObject>()
        .Where(part => NodeString(part["type"]) == "text").Select(part => NodeString(part["text"])));

    private static void Rename(JsonObject value, string source, string target, bool overwrite = false)
    {
        if (value.Remove(source, out var field) && (overwrite || !value.ContainsKey(target))) value[target] = field;
    }
}
