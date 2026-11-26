using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class AddisAIProviderRegistration
{
    public static IServiceCollection AddAddisAI(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, AddisAIProvider.Id, "https://api.addisassistant.com/api/");
        return services.AddProvider<AddisAIProvider>(AddisAIProvider.Id);
    }
}

internal sealed partial class AddisAIProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : IChatCompletionsProvider, IModelListingProvider, INativeChatOptionsProvider
{
    internal const string Id = "addisai";
    private const string ChatModel = "addis-1-alef";
    private static readonly Dictionary<string, string> Languages = new(StringComparer.OrdinalIgnoreCase)
    {
        ["am"] = "Amharic", ["om"] = "Afan Oromo", ["en"] = "English"
    };
    private static readonly HashSet<string> ChatControls = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "messages", "input", "prompt", "conversation_history", "target_language", "stream", "stream_options",
        "temperature", "top_p", "max_tokens", "max_completion_tokens", "store", "background", "previous_response_id", "conversation"
    };
    private static readonly HashSet<string> TranslationControls = new(ChatControls, StringComparer.OrdinalIgnoreCase)
    {
        "text", "source_language", "generation_config"
    };
    private static readonly HashSet<string> BufferedControls = new(StringComparer.OrdinalIgnoreCase) { "stream", "stream_options" };

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        await Task.CompletedTask;
        ct.ThrowIfCancellationRequested();
        yield return Model(ChatModel, "Addis 1 Alef", "language");
        foreach (var key in new[] { "am", "om" })
        {
            ct.ThrowIfCancellationRequested();
            yield return Model($"tts/{key}", $"AddisAI Text to Speech ({Languages[key]})", "speech");
            yield return Model($"stt/{key}", $"AddisAI Speech to Text ({Languages[key]})", "transcription");
        }
        foreach (var (source, sourceName) in Languages)
            foreach (var (target, targetName) in Languages)
            {
                ct.ThrowIfCancellationRequested();
                if (source != target)
                    yield return Model($"translate/{source}/{target}", $"AddisAI Translate {sourceName} to {targetName}", "language", ["translate", source, target]);
            }

        static Model Model(string model, string name, string type, string[]? tags = null)
            => new() { Id = model, Name = name, OwnedBy = "AddisAI", Type = type, Tags = tags };
    }

    public async Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (request.Tools?.Any() == true) throw new ProviderRequestException("AddisAI does not support chat tools.");
        var messages = request.Messages.Select(message => (message.Role, Text: ChatMessageContentExtensions.ToText(message.Content)))
            .Where(message => !string.IsNullOrWhiteSpace(message.Text)).ToList();
        var model = request.Model.ToLowerInvariant();
        using var client = clients.CreateClient(Id);
        var key = RequiredKey(credentials, Id);
        string text;
        JsonElement raw;
        ChatCompletionUsage? usage;
        var finish = "stop";
        if (model.StartsWith("translate/", StringComparison.Ordinal))
        {
            var parts = model.Split('/');
            if (parts.Length != 3 || !Languages.ContainsKey(parts[1]) || !Languages.ContainsKey(parts[2]) || parts[1] == parts[2])
                throw new ProviderRequestException("Use an AddisAI translation model in the form translate/{source}/{target}.", 422);
            var inputs = messages.Where(message => !string.Equals(message.Role, "assistant", StringComparison.OrdinalIgnoreCase))
                .Select(message => message.Text!).ToArray();
            if (inputs.Length == 0 || inputs.Any(input => input.Length > 2000))
                throw new ProviderRequestException("AddisAI translation requires text inputs of at most 2000 characters.");
            var payload = Options(request.AdditionalProperties, TranslationControls);
            payload["source_language"] = parts[1];
            payload["target_language"] = parts[2];
            var translations = new List<string>();
            var responses = new List<JsonElement>();
            usage = null;
            foreach (var input in inputs)
            {
                payload["text"] = input;
                var result = await PostAsync(client, "v1/translate", payload, key, headers, ct);
                var data = Property(result, "data");
                translations.Add(RequiredText(data, "translation"));
                responses.Add(result);
                if (Usage(data) is not { } item) continue;
                usage ??= new();
                usage.PromptTokens += item.PromptTokens;
                usage.CompletionTokens += item.CompletionTokens;
                usage.TotalTokens += item.TotalTokens;
            }
            raw = responses.Count == 1 ? responses[0] : JsonSerializer.SerializeToElement(new { responses }, Json);
            text = string.Join('\n', translations);
        }
        else
        {
            if (model != ChatModel) throw new ProviderRequestException("Unsupported AddisAI chat model.", 422);
            var lastUser = messages.FindLastIndex(message => string.Equals(message.Role, "user", StringComparison.OrdinalIgnoreCase));
            if (lastUser < 0) throw new ProviderRequestException("AddisAI chat requires a user text message.");
            var payload = Options(request.AdditionalProperties, ChatControls);
            Dictionary<string, object?> config = new(StringComparer.Ordinal);
            if (payload.Remove("generation_config", out var value))
            {
                if (value is not JsonElement { ValueKind: JsonValueKind.Object } supplied)
                    throw new ProviderRequestException("AddisAI generation_config must be an object.");
                config = Options(supplied.EnumerateObject().ToDictionary(field => field.Name, field => field.Value), BufferedControls);
            }
            if (request.Temperature is { } temperature) config["temperature"] = temperature;
            if ((request.MaxCompletionTokens ?? request.MaxTokens) is { } max) config["maxOutputTokens"] = max;
            if (request.TopP is { } topP) config["topP"] = topP;
            config["stream"] = false;
            payload["prompt"] = messages[lastUser].Text;
            payload["target_language"] = "am";
            payload["conversation_history"] = messages.Take(lastUser).Select(message => new
            {
                role = string.Equals(message.Role, "assistant", StringComparison.OrdinalIgnoreCase) ? "assistant" : "user",
                content = message.Text
            }).ToArray();
            payload["generation_config"] = config;
            raw = await PostAsync(client, "v1/chat_generate", payload, key, headers, ct);
            text = RequiredText(raw, "response_text");
            usage = Usage(raw);
            finish = String(raw, "finish_reason") switch { "length" or "MAX_TOKENS" => "length", "content_filter" or "SAFETY" => "content_filter", _ => "stop" };
        }

        return new()
        {
            Id = $"chatcmpl_{Guid.NewGuid():N}", Model = request.Model, Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            Choices = [new() { FinishReason = finish, Message = new() { Role = "assistant", Content = text } }],
            Usage = usage, AdditionalProperties = new()
            {
                ["providerMetadata"] = JsonSerializer.SerializeToElement(NativeMediaOptions.Metadata(Id, raw), Json)
            }
        };
    }

    public async IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        // AddisAI returns one JSON result, even when the client requests an SSE response.
        var result = await CompleteChatAsync(request, headers, ct);
        var choice = result.Choices.Single();
        yield return new()
        {
            Id = result.Id, Created = result.Created, Model = result.Model,
            Choices = [new() { Delta = new() { Role = "assistant", Content = choice.Message.Content } }]
        };
        ct.ThrowIfCancellationRequested();
        yield return new()
        {
            Id = result.Id, Created = result.Created, Model = result.Model, AdditionalProperties = result.AdditionalProperties,
            Choices = [new() { Delta = new(), FinishReason = choice.FinishReason }]
        };
        if (request.StreamOptions?.IncludeUsage == true && result.Usage is not null)
            yield return new() { Id = result.Id, Created = result.Created, Model = result.Model, Usage = result.Usage };
    }

    private static Dictionary<string, object?> Options(IReadOnlyDictionary<string, JsonElement>? extensions, IReadOnlySet<string> controls)
        => NativeMediaOptions.Read(Id, extensions, controls).ToDictionary(field => field.Key, field => (object?)field.Value, StringComparer.Ordinal);

    private static Task<JsonElement> PostAsync(HttpClient client, string endpoint, object data, string key, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => NativeHttp.PostAsync<object, JsonElement>(client, endpoint, data, outgoing => outgoing.TryAddWithoutValidation("X-API-Key", key), headers, Json, ct);

    private static string RequiredText(JsonElement element, string property)
        => String(element, property) is { } text && !string.IsNullOrWhiteSpace(text) ? text : throw new JsonException($"AddisAI did not return '{property}'.");

    private static ChatCompletionUsage? Usage(JsonElement response)
    {
        if (Property(response, "usage_metadata") is not { ValueKind: JsonValueKind.Object } usage) return null;
        return new() { PromptTokens = Count("prompt_token_count"), CompletionTokens = Count("candidates_token_count"), TotalTokens = Count("total_token_count") };

        int Count(string name) => Property(usage, name) is { ValueKind: JsonValueKind.Number } field && field.TryGetInt32(out var value) ? value : 0;
    }
}
