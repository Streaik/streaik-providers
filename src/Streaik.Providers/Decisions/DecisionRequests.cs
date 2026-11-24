using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.OpenAI.Models.Decisions;
using Streaik.Vercel.Models;

namespace Streaik.Providers.Decisions;

public static class DecisionRequests
{
    internal static readonly JsonSerializerOptions Json = JsonSerializerOptions.Web;
    private static readonly HashSet<string> Reserved = new(StringComparer.OrdinalIgnoreCase)
        { "model", "input", "state", "questions", "safety_identifier", "safetyIdentifier", "prompt_cache_key", "prompt_cache_options", "stream" };

    public static void Validate(OpenAIDecisionRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Model) || request.Input is null || request.Questions is not { Count: > 0 })
            throw new ProviderRequestException("Decisions requires a model, input, and at least one question.");
        if (request.AdditionalProperties?.Keys.Any(Reserved.Contains) == true)
            throw new ProviderRequestException("Decision options cannot override request fields or enable streaming.");
        foreach (var question in request.Questions)
        {
            if (question?.InstructionContent is null || question.InstructionContent.Text is { } instructions && string.IsNullOrWhiteSpace(instructions))
                throw new ProviderRequestException("Every decision question requires instructions.");
            switch (question)
            {
                case OpenAIDecisionPredicateQuestion: break;
                case OpenAIDecisionChoiceQuestion choice when choice.Choices is { Count: > 0 }
                    && choice.Choices.All(x => x is not null && IsChoice(x.Value.Value))
                    && choice.Choices.Select(x => ChoiceKey(x.Value)).Distinct(StringComparer.Ordinal).Count() == choice.Choices.Count: break;
                case OpenAIDecisionScoreQuestion score when score.Levels is { Count: > 0 }
                    && score.Levels.All(x => !string.IsNullOrWhiteSpace(x?.Label))
                    && score.Levels.Select(x => x.Label).Distinct(StringComparer.Ordinal).Count() == score.Levels.Count: break;
                default: throw new ProviderRequestException("Decision choices and score levels must be nonempty and distinct.");
            }
        }
    }

    internal static void Validate(DecisionRequest request, bool requiresInstructions = true)
    {
        if (string.IsNullOrWhiteSpace(request.Model) || request.State is null || request.Questions is not { Count: > 0 })
            throw new ProviderRequestException("Decisions requires a model, state, and at least one question.");
        if (request.StateFormat is not (null or "sdk"))
            throw new ProviderRequestException("stateFormat must be omitted or 'sdk'.");
        if (request.MaxOutputTokens is not null || request.Reasoning is not null)
            throw new ProviderRequestException("This Decisions provider does not support maxOutputTokens or reasoning.");
        foreach (var (id, question) in request.Questions)
        {
            if (string.IsNullOrWhiteSpace(id) || question is null || requiresInstructions && question.Instructions is null)
                throw new ProviderRequestException("Every decision question requires a nonempty ID and instructions.");
            if (question is not (DecisionBooleanQuestion or DecisionChoiceQuestion { Criteria.Count: > 0 } or DecisionScoreQuestion { Criteria.Count: > 0 }))
                throw new ProviderRequestException($"Unsupported question type or invalid criteria for '{id}'.");
        }
    }

    internal static OpenAIDecisionRequest ToOpenAI(DecisionRequest request, string provider, out JsonElement? rawInput)
    {
        Validate(request);
        var state = State(request);
        rawInput = null;
        OpenAIDecisionInput? input = null;
        if (state.ValueKind == JsonValueKind.Array && state.GetArrayLength() > 0 && state.EnumerateArray().All(IsLegacyMessage))
        {
            try { input = state.Deserialize<OpenAIDecisionInput>(Json); }
            catch (JsonException) when (request.StateFormat is null) { } // Legacy structured evidence can always be represented as JSON text.
            if (input is not null) rawInput = state;
        }
        var mapped = new OpenAIDecisionRequest
        {
            Model = request.Model,
            Input = input ?? new(state.ValueKind == JsonValueKind.String ? state.GetString()! : state.GetRawText()),
            Questions = request.Questions.Select(entry => MapQuestion(entry.Key, entry.Value)).ToArray()
        };
        foreach (var option in Options(request, provider))
        {
            switch (option.Name.ToLowerInvariant())
            {
                case "safety_identifier":
                case "safetyidentifier":
                    mapped.SafetyIdentifier = OptionString(option, allowNull: true);
                    break;
                case "prompt_cache_key":
                    mapped.PromptCacheKey = OptionString(option, allowNull: true);
                    break;
                case "prompt_cache_options":
                    try { mapped.PromptCacheOptions = option.Value.Deserialize<OpenAIDecisionPromptCacheOptions>(Json); }
                    catch (JsonException) { throw new ProviderRequestException("Invalid prompt_cache_options."); }
                    break;
                default:
                    if (Reserved.Contains(option.Name)) throw new ProviderRequestException("Decision options cannot override request fields or enable streaming.");
                    (mapped.AdditionalProperties ??= new(StringComparer.Ordinal))[option.Name] = option.Value;
                    break;
            }
        }
        Validate(mapped);
        return mapped;
    }

    internal static IEnumerable<JsonProperty> Options(DecisionRequest request, string provider)
    {
        if (request.ProviderOptions?.TryGetValue(provider, out var options) != true) yield break;
        if (options.ValueKind != JsonValueKind.Object) throw new ProviderRequestException($"providerOptions.{provider} must be an object.");
        foreach (var property in options.EnumerateObject()) yield return property;
    }

    internal static JsonElement State(DecisionRequest request, bool allowImages = true)
    {
        var state = request.State.Value;
        if (request.StateFormat is null) return state;
        if (state.ValueKind != JsonValueKind.Array || state.GetArrayLength() == 0)
            throw new ProviderRequestException("SDK decision state must be a nonempty array of parts.");
        var content = new JsonArray();
        foreach (var part in state.EnumerateArray())
        {
            if (part.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("SDK decision state requires typed parts.");
            if (part.TryGetProperty("providerOptions", out var options)
                && (options.ValueKind != JsonValueKind.Object || options.EnumerateObject().Any()))
                throw new ProviderRequestException("Decision state-part providerOptions are not supported.");
            switch (RequiredString(part, "type"))
            {
                case "text" when Only(part, "type", "text", "providerOptions"):
                    AddText(content, RequiredString(part, "text"));
                    break;
                case "json" when Only(part, "type", "value") && part.TryGetProperty("value", out var value):
                    AddText(content, value.GetRawText());
                    break;
                case "file" when Only(part, "type", "data", "mediaType", "filename", "providerOptions"):
                    AddFile(content, part, allowImages);
                    break;
                default: throw new ProviderRequestException("Unsupported SDK decision state part or fields.");
            }
        }
        return JsonSerializer.SerializeToElement(new JsonArray(new JsonObject { ["role"] = "user", ["content"] = content }), Json);
    }

    private static void AddFile(JsonArray content, JsonElement part, bool allowImages)
    {
        var media = RequiredString(part, "mediaType");
        if (!part.TryGetProperty("data", out var data) || data.ValueKind != JsonValueKind.Object)
            throw new ProviderRequestException("SDK file parts require tagged data.");
        var type = RequiredString(data, "type");
        if (part.TryGetProperty("filename", out _)) AddText(content, $"File: {RequiredString(part, "filename")}");
        if (type == "text" && Only(data, "type", "text")
            && (media is "text" or "application/json" || media.StartsWith("text/", StringComparison.OrdinalIgnoreCase)))
        {
            AddText(content, RequiredString(data, "text"));
            return;
        }
        if (!allowImages || !media.StartsWith("image/", StringComparison.OrdinalIgnoreCase))
            throw new ProviderRequestException("This Decisions provider does not support the SDK file media type.");
        var prefix = $"data:{media};base64,";
        var url = type switch
        {
            "data" when Only(data, "type", "data") => prefix + RequiredString(data, "data"),
            "url" when Only(data, "type", "url", "originalUrl") => RequiredString(data, data.TryGetProperty("originalUrl", out _) ? "originalUrl" : "url"),
            _ => throw new ProviderRequestException("Unsupported SDK decision file data.")
        };
        if (!url.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) || !IsInlineImage(url))
            throw new ProviderRequestException("SDK Decisions supports only valid inline base64 image URLs.");
        content.Add(new JsonObject { ["type"] = "input_image", ["image_url"] = url });
    }

    internal static OpenAIDecisionQuestion MapQuestion(string id, DecisionQuestion question)
    {
        OpenAIDecisionQuestion result = question switch
        {
            DecisionBooleanQuestion { Criteria: null } => new OpenAIDecisionPredicateQuestion(),
            DecisionBooleanQuestion boolean => new OpenAIDecisionChoiceQuestion { Choices =
                [new() { Value = true, Description = Text(boolean.Criteria!.True) }, new() { Value = false, Description = Text(boolean.Criteria.False) }] },
            DecisionChoiceQuestion choice => new OpenAIDecisionChoiceQuestion { Choices = choice.Criteria
                .Select(x => new OpenAIDecisionChoice { Value = x.Key, Description = Text(x.Value) }).ToArray() },
            DecisionScoreQuestion score => new OpenAIDecisionScoreQuestion { Levels = score.Criteria
                .Select((x, i) => new OpenAIDecisionScoreLevel { Label = Index(i), Description = Text(x) }).ToArray() },
            _ => throw new ProviderRequestException("Unsupported decision question type.")
        };
        result.Name = id;
        result.Instructions = Text(question.Instructions)!;
        return result;
    }

    internal static string Index(int index) => index.ToString(CultureInfo.InvariantCulture);
    internal static bool IsChoice(JsonElement value) => value.ValueKind is JsonValueKind.String or JsonValueKind.True or JsonValueKind.False;
    internal static string ChoiceKey(OpenAIDecisionChoiceValue value) => value.Value.ValueKind == JsonValueKind.String
        ? "string:" + value.Value.GetString() : "boolean:" + value.Value.GetRawText();
    private static string? Text(DecisionInput? input) => input is null ? null : input.Value.ValueKind == JsonValueKind.String
        ? input.Value.GetString() : input.Value.GetRawText();
    private static string? OptionString(JsonProperty option, bool allowNull) => option.Value.ValueKind == JsonValueKind.String
        || allowNull && option.Value.ValueKind == JsonValueKind.Null ? option.Value.GetString()
        : throw new ProviderRequestException($"'{option.Name}' must be a string or null.");
    private static string RequiredString(JsonElement root, string name) => root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
        ? value.GetString()! : throw new ProviderRequestException($"SDK decision part requires string '{name}'.");
    private static bool Only(JsonElement value, params string[] names) => value.ValueKind == JsonValueKind.Object
        && value.EnumerateObject().All(x => names.Contains(x.Name, StringComparer.Ordinal));
    private static void AddText(JsonArray content, string text) => content.Add(new JsonObject { ["type"] = "input_text", ["text"] = text });

    private static bool IsLegacyMessage(JsonElement value) => Only(value, "role", "type", "content")
        && value.TryGetProperty("role", out var role) && role.ValueKind == JsonValueKind.String && role.GetString() == "user"
        && (!value.TryGetProperty("type", out var type) || type.ValueKind == JsonValueKind.String && type.GetString() == "message")
        && value.TryGetProperty("content", out var content) && (content.ValueKind == JsonValueKind.String
            || content.ValueKind == JsonValueKind.Array && content.GetArrayLength() > 0 && content.EnumerateArray().All(IsLegacyPart));

    private static bool IsLegacyPart(JsonElement part) => part.ValueKind == JsonValueKind.Object
        && part.TryGetProperty("type", out var type) && type.ValueKind == JsonValueKind.String && type.GetString() switch
        {
            "input_text" => Only(part, "type", "text") && part.TryGetProperty("text", out var text) && text.ValueKind == JsonValueKind.String,
            "input_image" => Only(part, "type", "image_url", "detail") && part.TryGetProperty("image_url", out var url)
                && url.ValueKind == JsonValueKind.String && IsInlineImage(url.GetString()!),
            _ => false
        };

    private static bool IsInlineImage(string url)
    {
        var comma = url.IndexOf(',');
        if (!url.StartsWith("data:image/", StringComparison.OrdinalIgnoreCase) || comma <= "data:image/;base64".Length
            || !url.AsSpan(0, comma).EndsWith(";base64", StringComparison.OrdinalIgnoreCase)) return false;
        try { return Convert.FromBase64String(url[(comma + 1)..]).Length > 0; }
        catch (FormatException) { return false; }
    }
}
