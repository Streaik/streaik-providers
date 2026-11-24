using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class CohereProviderRegistration
{
    public static IServiceCollection AddCohere(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, CohereProvider.Id, "https://api.cohere.com/");
        return services.AddProvider<CohereProvider>(CohereProvider.Id);
    }
}

internal sealed partial class CohereProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IModelListingProvider, IChatCompletionsProvider, INativeChatOptionsProvider
{
    internal const string Id = "cohere";

    private HttpClient Client() => clients.CreateClient(Id);
    private Action<HttpRequestHeaders> Auth() => Bearer(RequiredKey(credentials, Id));
    private static Action<HttpRequestHeaders> Bearer(string key) => h => h.Authorization = new("Bearer", key);
    private static int? Int(JsonElement value, string key) => Property(value, key) is { ValueKind: JsonValueKind.Number } number
        && number.TryGetInt32(out var count) ? count : null;
    private static Dictionary<string, JsonElement> Scoped(JsonElement raw) => new() { [Id] = raw };
    private static JsonElement Metadata(JsonElement raw) => JsonSerializer.SerializeToElement(Scoped(raw), Json);

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (credentials.GetApiKey(Id) is not { } key || string.IsNullOrWhiteSpace(key)) yield break;
        using var client = Client();
        var tokens = new HashSet<string>(StringComparer.Ordinal);
        var models = new HashSet<string>(StringComparer.Ordinal);
        string? token = null;
        do
        {
            var page = await NativeHttp.GetAsync<JsonElement>(client, "v1/models?page_size=1000"
                + (token is null ? "" : "&page_token=" + Uri.EscapeDataString(token)), Bearer(key), ct);
            foreach (var model in Array(Property(page, "models")))
            {
                if (String(model, "name") is not { Length: > 0 } name || !models.Add(name)) continue;
                var capabilities = Array(Property(model, "features")).Concat(Array(Property(model, "endpoints")))
                    .Where(x => x.ValueKind == JsonValueKind.String).Select(x => x.GetString()!).ToArray();
                var type = IsTranscriptionModel(name) || capabilities.Any(IsTranscriptionModel) ? "transcription"
                    : name.Contains("rerank", StringComparison.OrdinalIgnoreCase) || capabilities.Contains("rerank") ? "reranking"
                    : name.Contains("embed", StringComparison.OrdinalIgnoreCase) || capabilities.Contains("embed") ? "embedding" : "language";
                var translate = name.Equals("command-a-translate", StringComparison.OrdinalIgnoreCase)
                    || name.StartsWith("command-a-translate-", StringComparison.OrdinalIgnoreCase);
                yield return new Model { Id = name, Name = name, Type = type, Tags = translate ? ["translate"] : null, OwnedBy = Id,
                    ContextWindow = Int(model, "context_length"), Description = String(model, "description") };
            }
            token = String(page, "next_page_token");
        } while (!string.IsNullOrEmpty(token) && tokens.Add(token));
    }

    public async Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = Client();
        if (IsParseModel(request.Model)) return await ParseAsync(client, request, headers, ct);
        var root = await NativeHttp.PostAsync<JsonObject, JsonElement>(client, "v2/chat", ChatBody(request, false), Auth(), headers, Json, ct);
        var message = Property(root, "message");
        if (message.ValueKind != JsonValueKind.Object) throw new JsonException("Cohere returned no chat message.");
        var content = Property(message, "content");
        var text = content.ValueKind == JsonValueKind.String ? content.GetString() : JoinContent(content, "text");
        var reasoning = JoinContent(content, "thinking") + String(message, "tool_plan");
        var tools = Array(Property(message, "tool_calls")).Select(x => Tool(x, null)).ToArray();
        var annotations = Annotations(Property(message, "citations"));
        return Completion(String(root, "id"), String(root, "model") ?? request.Model, text, reasoning, tools, annotations,
            Finish(String(root, "finish_reason"), tools.Length > 0), Usage(Property(root, "usage")), root);
    }

    public async IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = Client();
        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var id = $"chatcmpl_{Guid.NewGuid():N}";
        var model = request.Model;
        ChatCompletionUpdate Chunk(object delta, string? finish = null, object? usage = null, JsonElement? raw = null)
            => Wire<ChatCompletionUpdate>(new { id, created, model, @object = "chat.completion.chunk", choices = new[] { new { index = 0, delta, finish_reason = finish } },
                usage, providerMetadata = raw is { } data ? Metadata(data) : (JsonElement?)null });

        if (IsParseModel(model))
        {
            var result = await ParseAsync(client, request, headers, ct);
            var choice = result.Choices!.Single();
            id = result.Id!;
            yield return Chunk(new { role = "assistant", content = choice.Message?.Content });
            yield return Chunk(new { }, choice.FinishReason, result.Usage, result.AdditionalProperties!["providerMetadata"].GetProperty(Id));
            yield break;
        }

        var sawTool = false;
        var ended = false;
        var citations = new List<JsonElement>();
        await foreach (var part in NativeHttp.StreamAsync<JsonObject, JsonElement>(client, "v2/chat", ChatBody(request, true), Auth(), headers, Json,
            e => String(e, "type") is "message-end" or "error", ct))
        {
            var delta = Property(part, "delta");
            var message = Property(delta, "message");
            var type = String(part, "type");
            var index = Int(part, "index") ?? 0;
            if (String(part, "id") is { } upstreamId) id = upstreamId;
            if (String(part, "model") is { } upstreamModel) model = upstreamModel;
            switch (type)
            {
                case "message-start":
                    yield return Chunk(new { role = "assistant" });
                    break;
                case "content-start":
                case "content-delta":
                    var content = Property(message, "content");
                    var text = String(content, "text");
                    var thinking = String(content, "thinking");
                    if (text is not null || thinking is not null) yield return Chunk(new { content = text, reasoning_content = thinking });
                    break;
                case "tool-plan-delta":
                    if (String(message, "tool_plan") is { } plan) yield return Chunk(new { reasoning_content = plan });
                    break;
                case "tool-call-start":
                case "tool-call-delta":
                    sawTool = true;
                    var tool = Property(message, "tool_calls");
                    if (tool.ValueKind == JsonValueKind.Array) tool = Array(tool).FirstOrDefault();
                    yield return Chunk(new { tool_calls = new[] { Tool(tool, index) } });
                    break;
                case "citation-start":
                case "citation-end":
                    if (Property(message, "citations") is { ValueKind: JsonValueKind.Object or JsonValueKind.Array } citation)
                    {
                        citations.Add(citation);
                        if (type == "citation-start" && Annotations(citation) is { Length: > 0 } annotations)
                            yield return Chunk(new { annotations });
                    }
                    break;
                case "message-end":
                    ended = true;
                    var termination = String(delta, "finish_reason") ?? String(part, "finish_reason")
                        ?? throw new JsonException("Cohere ended the stream without a finish reason.");
                    var usage = Property(delta, "usage");
                    if (usage.ValueKind == JsonValueKind.Undefined) usage = Property(part, "usage");
                    var metadata = ToObject(part);
                    if (citations.Count > 0) metadata["citations"] = JsonSerializer.SerializeToNode(citations, Json);
                    yield return Chunk(new { }, Finish(termination, sawTool), Usage(usage), JsonSerializer.SerializeToElement(metadata, Json));
                    break;
                case "error": throw new IOException("Cohere returned a chat stream error.");
            }
        }
        if (!ended) throw new IOException("Cohere ended the stream before message-end.");
    }

    private static JsonObject ChatBody(ChatCompletionOptions request, bool stream)
    {
        var input = JsonSerializer.SerializeToElement(request, Json);
        var payload = CopyOptions(request.AdditionalProperties, "model", "messages", "stream", "max_completion_tokens", "top_p", "n", "store", "stream_options");
        ObjectOption(payload, "thinking", "type");
        ObjectOption(payload, "citation_options", "mode");
        payload["model"] = request.Model;
        payload["stream"] = stream;
        if (Int(input, "n") is > 1) throw new ProviderRequestException("Cohere supports one choice per request.");
        var messages = new JsonArray();
        foreach (var original in Array(Property(input, "messages")))
        {
            var role = String(original, "role");
            if (role is not ("developer" or "system" or "user" or "assistant" or "tool")) throw new ProviderRequestException("Unsupported Cohere message role.");
            var message = new JsonObject { ["role"] = role == "developer" ? "system" : role };
            var content = Property(original, "content");
            if (content.ValueKind == JsonValueKind.String) message["content"] = content.GetString();
            else if (content.ValueKind == JsonValueKind.Array)
            {
                if (Array(content).Any(p => String(p, "type") is not ("text" or "thinking" or "image_url")))
                    throw new ProviderRequestException("Cohere chat accepts text and image content parts.", 422);
                message["content"] = Copy(content);
            }
            foreach (var field in new[] { "tool_calls", "tool_call_id", "tool_plan" })
                if (original.TryGetProperty(field, out var value)) message[field] = Copy(value);
            messages.Add(message);
        }
        payload["messages"] = messages;
        if ((Int(input, "max_completion_tokens") ?? Int(input, "max_tokens")) is { } max) payload["max_tokens"] = max;
        if (input.TryGetProperty("top_p", out var topP)) payload["p"] = Copy(topP);
        foreach (var name in new[] { "temperature", "frequency_penalty", "presence_penalty", "seed", "logprobs" })
            if (input.TryGetProperty(name, out var value)) payload[name] = Copy(value);
        if (input.TryGetProperty("stop", out var stop)) payload["stop_sequences"] = stop.ValueKind == JsonValueKind.String ? new JsonArray(JsonValue.Create(stop.GetString())) : Copy(stop);
        if (Property(input, "tools") is { ValueKind: JsonValueKind.Array } tools && tools.GetArrayLength() > 0)
        {
            if (Array(tools).Any(tool => String(tool, "type") != "function")) throw new ProviderRequestException("Cohere chat supports function tools only.");
            var native = (JsonArray)Copy(tools)!;
            var strict = false;
            foreach (var tool in native)
                if (tool?["function"] is JsonObject fn && fn.Remove("strict", out var node) && node?.GetValue<bool>() == true) strict = true;
            payload["tools"] = native;
            if (strict && !payload.ContainsKey("strict_tools")) payload["strict_tools"] = true;
        }
        if (request.ToolChoiceOption is not null && input.TryGetProperty("tool_choice", out var choice))
        {
            if (choice.ValueKind == JsonValueKind.Object)
            {
                var name = String(Property(choice, "function"), "name");
                if (name is null || payload["tools"] is not JsonArray available
                    || available.OfType<JsonObject>().Where(t => (string?)t["function"]?["name"] == name).Take(2).ToArray() is not [var selected])
                    throw new ProviderRequestException("Selected Cohere function is not defined.");
                payload["tools"] = new JsonArray(selected.DeepClone());
                payload["tool_choice"] = "REQUIRED";
            }
            else if (choice.ValueKind == JsonValueKind.String && choice.GetString() is { } mode)
                payload["tool_choice"] = mode.ToLowerInvariant() switch { "required" => "REQUIRED", "none" => "NONE", "auto" => null, _ => mode };
        }
        if (request.ResponseFormat is not null && Property(input, "response_format") is { ValueKind: JsonValueKind.Object } format)
        {
            var type = String(format, "type");
            if (type == "json_schema") payload["response_format"] = new JsonObject { ["type"] = "json_object", ["json_schema"] = Copy(Property(Property(format, "json_schema"), "schema")) };
            else if (type != "text") payload["response_format"] = Copy(format);
        }
        return payload;
    }

    private static void ObjectOption(JsonObject payload, string name, string defaultField)
    {
        if (payload[name] is null) return;
        if (payload[name] is not JsonObject option) throw new ProviderRequestException($"Cohere '{name}' must be an object.");
        if (!option.ContainsKey(defaultField)) option[defaultField] = "enabled";
    }

    private static JsonObject CopyOptions(IEnumerable<KeyValuePair<string, JsonElement>>? options, params string[] reserved)
    {
        var body = new JsonObject();
        foreach (var (name, value) in options ?? [])
            if (!reserved.Contains(name, StringComparer.OrdinalIgnoreCase)) body[name] = Copy(value);
        return body;
    }

    private static T Wire<T>(object data) => JsonSerializer.SerializeToElement(data, Json).Deserialize<T>(Json)!;
    private static string JoinContent(JsonElement content, string field) => string.Concat(Array(content).Select(p => String(p, field)));
    private static object Tool(JsonElement tool, int? index) => new { index, id = String(tool, "id"), type = String(tool, "type") ?? "function",
        function = new { name = String(Property(tool, "function"), "name"), arguments = String(Property(tool, "function"), "arguments") } };
    private static object[] Annotations(JsonElement citations)
        => (citations.ValueKind == JsonValueKind.Object ? [citations] : Array(citations)).SelectMany(c => Array(Property(c, "sources")).Select(source =>
        {
            var document = Property(source, "document");
            return new { start = Int(c, "start"), end = Int(c, "end"), url = String(source, "url") ?? String(document, "url"),
                title = String(source, "title") ?? String(document, "title") };
        })).Where(a => a.start.HasValue && a.end.HasValue && Uri.TryCreate(a.url, UriKind.Absolute, out var url) && url.Scheme is "http" or "https")
        .Select(a => (object)new { type = "url_citation", url_citation = new { start_index = a.start, end_index = a.end, a.url, title = a.title ?? a.url } }).ToArray();

    private static object? Usage(JsonElement usage)
    {
        var tokens = Property(usage, "tokens");
        var billed = Property(usage, "billed_units");
        var input = Int(tokens, "input_tokens") ?? Int(billed, "input_tokens");
        var output = Int(tokens, "output_tokens") ?? Int(billed, "output_tokens");
        if (!input.HasValue && !output.HasValue) return null;
        var cached = Int(tokens, "cached_input_tokens") ?? Int(usage, "cached_tokens");
        var reasoning = Int(tokens, "reasoning_tokens");
        return new { prompt_tokens = input ?? 0, completion_tokens = output ?? 0, total_tokens = Int(tokens, "total_tokens") ?? (input ?? 0) + (output ?? 0),
            prompt_tokens_details = cached.HasValue ? new { cached_tokens = cached } : null,
            completion_tokens_details = reasoning.HasValue ? new { reasoning_tokens = reasoning } : null };
    }

    private static string Finish(string? value, bool tool) => value?.ToUpperInvariant() switch
    {
        "MAX_TOKENS" or "MAX_TOKENS_REACHED" => "length",
        "TOOL_CALL" or "TOOL_CALLS" => "tool_calls",
        "ERROR" or "ERROR_TOXIC" or "ERROR_LIMIT" => throw new IOException("Cohere did not complete the request."),
        "SAFETY" => "content_filter",
        _ => tool ? "tool_calls" : "stop"
    };

    private static ChatCompletion Completion(string? id, string model, string? text, string? reasoning, object[] tools, object[] annotations,
        string finish, object? usage, JsonElement raw)
        => Wire<ChatCompletion>(new { id = id ?? $"chatcmpl_{Guid.NewGuid():N}", model, created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), @object = "chat.completion",
            choices = new[] { new { index = 0, message = new { role = "assistant", content = text, reasoning_content = string.IsNullOrEmpty(reasoning) ? null : reasoning,
                tool_calls = tools.Length > 0 ? tools : null, annotations = annotations.Length > 0 ? annotations : null }, finish_reason = finish } }, usage, providerMetadata = Metadata(raw) });

    private static bool IsParseModel(string model) => model.StartsWith("parse-", StringComparison.OrdinalIgnoreCase);

    private async Task<ChatCompletion> ParseAsync(HttpClient client, ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var messages = Array(Property(JsonSerializer.SerializeToElement(request, Json), "messages"));
        var documents = messages.Where(m => String(m, "role") == "user").SelectMany(m => Array(Property(m, "content"))).Select(part =>
        {
            if (String(part, "type") == "image_url") return String(Property(part, "image_url"), "url");
            if (String(part, "type") != "file") return null;
            var file = Property(part, "file");
            return String(file, "file_url") ?? String(file, "file_data");
        }).Where(value => !string.IsNullOrWhiteSpace(value)).ToArray();
        if (documents.Length == 0) throw new ProviderRequestException("Cohere Parse requires document attachments.", 422);
        var bodies = new List<JsonElement>();
        var markdown = new List<string>();
        var incomplete = false;
        var input = 0;
        var output = 0;
        foreach (var doc in documents)
        {
            var payload = CopyOptions(request.AdditionalProperties, "model", "document", "messages", "stream");
            payload["model"] = request.Model;
            payload["document"] = new JsonObject { ["type"] = "image_url", ["image_url"] = doc };
            payload["output_format"] ??= JsonValue.Create("markdown");
            var body = await NativeHttp.PostAsync<JsonObject, JsonElement>(client, "v2/parse", payload, Auth(), headers, Json, ct);
            bodies.Add(body);
            foreach (var page in Array(Property(body, "pages")))
                if (String(Property(page, "markdown"), "content") is { } text) markdown.Add(text);
            incomplete |= Finish(String(body, "finish_reason"), false) == "length";
            var tokens = Property(Property(body, "meta"), "tokens");
            input += Int(tokens, "input_tokens") ?? 0;
            output += Int(tokens, "output_tokens") ?? 0;
        }
        return Completion(null, request.Model, string.Join("\n\n", markdown), null, [], [], incomplete ? "length" : "stop",
            new { prompt_tokens = input, completion_tokens = output, total_tokens = input + output }, JsonSerializer.SerializeToElement(new { documents = bodies }, Json));
    }
}
