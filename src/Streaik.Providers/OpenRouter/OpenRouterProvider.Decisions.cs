using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.OpenAI.Models.Decisions;
using Streaik.Vercel.Models;
using Streaik.Providers.Decisions;

namespace Streaik.Providers.OpenRouter;

public sealed partial class OpenRouterProvider
{
    private sealed record RouterAnswer(DecisionAnswer Answer, double? Confidence);

    public async Task<DecisionResponse> DecideAsync(DecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        DecisionRequests.Validate(request);
        var warnings = new List<object>();
        var state = DecisionRequests.State(request, allowImages: false);
        var (body, upstreamHeaders, normalized, input, output) = await DecideRouterAsync(request.Model, state, request.Questions,
            DecisionRequests.Options(request, Id), warnings, headers, ct);
        return ToSdk(body, upstreamHeaders, normalized, input, output, warnings);
    }

    public async Task<OpenAIDecisionResponse> DecideAsync(OpenAIDecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        DecisionRequests.Validate(request);
        if (request.PromptCacheKey is not null || request.PromptCacheOptions is not null)
            throw new ProviderRequestException("OpenRouter Decisions does not support OpenAI prompt caching.");
        if (request.Input.Text is null && (request.Input.Items is not { Count: > 0 }
            || request.Input.Items.Any(x => x is not OpenAIDecisionInputMessage { Role: "user" } message
                || message.Content.Text is null && (message.Content.Parts is not { Count: > 0 }
                    || message.Content.Parts.Any(part => part is not OpenAIDecisionInputText { PromptCacheBreakpoint: null })))))
            throw new ProviderRequestException("OpenRouter Decisions supports only text input and user text messages on this route.");

        var questions = new Dictionary<string, DecisionQuestion>(StringComparer.Ordinal);
        for (var index = 0; index < request.Questions.Count; index++)
        {
            var original = request.Questions[index];
            var instructions = original.InstructionContent.Text
                ?? throw new ProviderRequestException("OpenRouter Decisions requires text question instructions.");
            DecisionQuestion mapped = original switch
            {
                OpenAIDecisionPredicateQuestion => new DecisionBooleanQuestion(),
                OpenAIDecisionChoiceQuestion choice => new DecisionChoiceQuestion { Criteria = choice.Choices.Select((x, i) =>
                    KeyValuePair.Create(OptionId(i), (DecisionInput?)new DecisionInput(Describe(x.Value.Value.ValueKind == JsonValueKind.String
                        ? x.Value.Value.GetString()! : x.Value.Value.GetRawText(), x.Description))))
                    .ToDictionary(x => x.Key, x => x.Value, StringComparer.Ordinal) },
                OpenAIDecisionScoreQuestion score => new DecisionScoreQuestion { Criteria = score.Levels
                    .Select(x => (DecisionInput?)new DecisionInput(Describe(x.Label, x.Description))).ToArray() },
                _ => throw new ProviderRequestException("Unsupported decision question type.")
            };
            mapped.Instructions = instructions;
            questions.Add(QuestionId(index), mapped);
        }

        var options = JsonSerializer.SerializeToElement(request.AdditionalProperties ?? [], DecisionRequests.Json);
        var state = JsonSerializer.SerializeToElement(request.Input, DecisionRequests.Json);
        var warnings = new List<object>();
        var (body, upstreamHeaders, normalized, input, output) = await DecideRouterAsync(request.Model, state, questions,
            options.EnumerateObject(), warnings, headers, ct);
        if ((long)input + output > int.MaxValue) throw new JsonException("Invalid decision usage total.");
        Warn(warnings, null, "OpenRouter does not report cached, cache-write, or reasoning tokens; OpenAI-compatible detail counters default to zero. Ordered scores use an evenly spaced [0,1] scale.");
        if (request.SafetyIdentifier is not null)
            Warn(warnings, "safety_identifier", "OpenRouter Decisions does not support safety_identifier; it was not forwarded or mapped to user.");
        foreach (var (id, answer) in normalized)
        {
            if (answer.Answer is DecisionBooleanAnswer) continue;
            if (answer.Confidence is null) Warn(warnings, id, "OpenRouter omitted confidence; the required OpenAI-compatible field defaults to zero.");
            if (answer.Answer is DecisionChoiceAnswer { Probabilities: null } or DecisionScoreAnswer { Probabilities: null })
                Warn(warnings, id, "OpenRouter omitted probabilities; the OpenAI-compatible list is empty.");
        }
        var sdk = ToSdk(body, upstreamHeaders, normalized, input, output, warnings);
        return new()
        {
            Model = DecisionResults.RequiredModel(body),
            Answers = request.Questions.Select((question, index) => ToOpenAIAnswer(question, normalized[QuestionId(index)])).ToArray(),
            Usage = new() { InputTokens = input, OutputTokens = output, TotalTokens = input + output },
            AdditionalProperties = new()
            {
                ["providerMetadata"] = JsonSerializer.SerializeToElement(sdk.ProviderMetadata, DecisionRequests.Json),
                ["response"] = JsonSerializer.SerializeToElement(sdk.Response, DecisionRequests.Json),
                ["warnings"] = JsonSerializer.SerializeToElement(warnings, DecisionRequests.Json)
            }
        };
    }

    private static DecisionResponse ToSdk(JsonElement body, Dictionary<string, string?> headers, Dictionary<string, RouterAnswer> answers,
        int input, int output, List<object> warnings)
    {
        var result = DecisionResults.Sdk(Id, DecisionResults.RequiredModel(body), body, headers,
            answers.ToDictionary(x => x.Key, x => x.Value.Answer, StringComparer.Ordinal), input, output, warnings);
        var cost = DecisionResults.Property(DecisionResults.Property(body, "usage"), "cost");
        if (cost.ValueKind == JsonValueKind.Number && cost.TryGetDecimal(out var value) && value >= 0)
            result.ProviderMetadata!["gateway"] = JsonSerializer.SerializeToElement(new { cost = value }, DecisionRequests.Json);
        return result;
    }

    private async Task<(JsonElement Body, Dictionary<string, string?> Headers, Dictionary<string, RouterAnswer> Answers, int Input, int Output)>
        DecideRouterAsync(string model, JsonElement state, Dictionary<string, DecisionQuestion> questions, IEnumerable<JsonProperty> options,
            List<object> warnings, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var mapped = new JsonObject();
        foreach (var (id, question) in questions)
        {
            var native = new JsonObject
            {
                ["type"] = question is DecisionBooleanQuestion ? "noul" : question.Type,
                ["instructions"] = Node(question.Instructions!.Value)
            };
            switch (question)
            {
                case DecisionBooleanQuestion { Criteria: { } criteria }:
                    native["criteria"] = new JsonObject { ["true"] = Guidance(criteria.True), ["false"] = Guidance(criteria.False) };
                    if (criteria.True is null || criteria.False is null) Warn(warnings, id, "Null or omitted boolean guidance is forwarded as an empty string.");
                    break;
                case DecisionChoiceQuestion choice:
                    native["criteria"] = JsonSerializer.SerializeToNode(choice.Criteria, DecisionRequests.Json);
                    break;
                case DecisionScoreQuestion score:
                    native["criteria"] = new JsonArray(score.Criteria.Select(Guidance).ToArray());
                    if (score.Criteria.Any(x => x is null)) Warn(warnings, id, "Null score guidance is forwarded as an empty string, preserving every level.");
                    break;
            }
            mapped.Add(id, native);
        }
        var payload = new JsonObject { ["model"] = model, ["state"] = Node(state), ["questions"] = mapped };
        foreach (var option in options) AddOption(payload, option, warnings);
        var key = MediaHttp.RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        var (body, upstreamHeaders) = await NativeHttp.PostJsonAsync(client, "alpha/decisions", payload,
            h => h.Authorization = new("Bearer", key), headers, DecisionRequests.Json, ct);
        DecisionResults.RequiredModel(body);
        var (input, output) = DecisionResults.Usage(body);
        var answers = DecisionResults.Property(body, "answers");
        if (answers.ValueKind != JsonValueKind.Object) throw new JsonException("OpenRouter decision answers must be an object.");
        var normalized = new Dictionary<string, RouterAnswer>(StringComparer.Ordinal);
        foreach (var entry in answers.EnumerateObject())
        {
            if (!questions.TryGetValue(entry.Name, out var question) || !normalized.TryAdd(entry.Name, ReadAnswer(question, entry.Value)))
                throw new JsonException("OpenRouter returned an unknown or duplicate decision answer.");
        }
        if (normalized.Count != questions.Count) throw new JsonException("OpenRouter did not answer every question.");
        return (body, upstreamHeaders, questions.Keys.ToDictionary(id => id, id => normalized[id], StringComparer.Ordinal), input, output);
    }

    private static RouterAnswer ReadAnswer(DecisionQuestion question, JsonElement answer)
    {
        if (DecisionResults.String(answer, "type") != (question is DecisionBooleanQuestion ? "noul" : question.Type))
            throw new JsonException("OpenRouter decision answer type does not match its question.");
        var detail = DecisionResults.Property(answer, "confidence");
        double? confidence = detail.ValueKind == JsonValueKind.Undefined ? null : DecisionResults.Probability(detail);
        var rawProbabilities = DecisionResults.Property(answer, "probabilities");
        switch (question)
        {
            case DecisionBooleanQuestion:
                return new(new DecisionBooleanAnswer { Probability = DecisionResults.Probability(DecisionResults.Property(answer, "noul")) }, confidence);
            case DecisionChoiceQuestion choice:
                var selected = DecisionResults.String(answer, "choice");
                if (selected is null || !choice.Criteria.ContainsKey(selected)) throw new JsonException("Unknown selected choice.");
                var distribution = ReadDistribution(rawProbabilities, choice.Criteria.Keys);
                if (distribution is not null && distribution[selected] + 1e-6 < distribution.Values.Max())
                    throw new JsonException("Selected choice is not a maximal option.");
                return new(new DecisionChoiceAnswer { Choice = selected, Probabilities = distribution }, confidence);
            case DecisionScoreQuestion score:
                var levels = Enumerable.Range(0, score.Criteria.Count).ToDictionary(DecisionRequests.Index);
                var probabilities = ReadDistribution(rawProbabilities, levels.Keys);
                var value = DecisionResults.Number(DecisionResults.Property(answer, "score"));
                if (value < 0 || value > levels.Count - 1 || probabilities is not null
                    && Math.Abs(value - probabilities.Sum(x => levels[x.Key] * x.Value)) > 1e-6)
                    throw new JsonException("Invalid OpenRouter weighted score.");
                var legend = DecisionResults.Property(answer, "legend");
                if (legend.ValueKind != JsonValueKind.Undefined && (legend.ValueKind != JsonValueKind.Object
                    || legend.EnumerateObject().Count() != levels.Count || levels.Keys.Any(key =>
                        DecisionResults.Property(legend, key).ValueKind is not (JsonValueKind.String or JsonValueKind.Object or JsonValueKind.Array))))
                    throw new JsonException("Invalid OpenRouter score legend.");
                return new(new DecisionScoreAnswer { Score = value, Probabilities = probabilities }, confidence);
            default: throw new JsonException("Unsupported OpenRouter decision question type.");
        }
    }

    private static Dictionary<string, double>? ReadDistribution(JsonElement value, IEnumerable<string> keys) => value.ValueKind switch
    {
        JsonValueKind.Undefined => null,
        JsonValueKind.Object => DecisionResults.Distribution(value.EnumerateObject().Select(x => (x.Name, DecisionResults.Probability(x.Value))), keys),
        _ => throw new JsonException("Expected an OpenRouter probability map.")
    };

    private static OpenAIDecisionAnswer ToOpenAIAnswer(OpenAIDecisionQuestion question, RouterAnswer result)
    {
        // Missing confidence/likelihood data is never inferred; the OpenAI shape requires these fields.
        var confidence = result.Confidence ?? 0;
        var mapped = (question, result.Answer) switch
        {
            (_, DecisionBooleanAnswer predicate) => (OpenAIDecisionAnswer)new OpenAIDecisionPredicateAnswer { Probability = predicate.Probability },
            (OpenAIDecisionChoiceQuestion choice, DecisionChoiceAnswer selected) => new OpenAIDecisionChoiceAnswer
            {
                Choice = choice.Choices[Enumerable.Range(0, choice.Choices.Count).Single(i => OptionId(i) == selected.Choice)].Value,
                Confidence = confidence,
                Probabilities = selected.Probabilities is not { } p ? [] : choice.Choices.Select((x, i) => new OpenAIDecisionChoiceProbability
                    { Value = x.Value, Probability = p[OptionId(i)] }).ToArray()
            },
            (OpenAIDecisionScoreQuestion score, DecisionScoreAnswer scored) => ScaleScore(score, scored, confidence),
            _ => throw new JsonException("Invalid OpenRouter decision answer.")
        };
        mapped.Name = question.Name;
        return mapped;
    }

    private static OpenAIDecisionScoreAnswer ScaleScore(OpenAIDecisionScoreQuestion question, DecisionScoreAnswer answer, double confidence)
    {
        var divisor = (double)Math.Max(1, question.Levels.Count - 1);
        return new()
        {
            Score = answer.Score / divisor,
            Confidence = confidence,
            Probabilities = answer.Probabilities is not { } p ? [] : question.Levels.Select((x, i) => new OpenAIDecisionScoreProbability
                { Label = x.Label, Value = i / divisor, Probability = p[DecisionRequests.Index(i)] }).ToArray()
        };
    }

    private static void AddOption(JsonObject payload, JsonProperty option, List<object> warnings)
    {
        if (new[] { "model", "state", "questions", "input", "stream" }.Contains(option.Name, StringComparer.OrdinalIgnoreCase))
            throw new ProviderRequestException("OpenRouter decision options cannot override request fields or enable streaming.");
        var kind = option.Value.ValueKind;
        var valid = option.Name switch
        {
            "provider" => kind is JsonValueKind.Object or JsonValueKind.Null,
            "trace" => kind == JsonValueKind.Object,
            "user" or "session_id" => kind == JsonValueKind.String && option.Value.GetString()!.Length <= 256,
            _ => false
        };
        if (valid) payload[option.Name] = Node(option.Value);
        else if (option.Name is "provider" or "trace" or "user" or "session_id")
            throw new ProviderRequestException($"Invalid or unsupported OpenRouter decision option '{option.Name}'.");
        else Warn(warnings, option.Name, "OpenRouter Decisions does not support this option; it was not forwarded.");
    }

    private static JsonNode? Node(JsonElement value) => JsonNode.Parse(value.GetRawText());
    private static JsonNode? Guidance(DecisionInput? input) => input is null ? JsonValue.Create("") : Node(input.Value);
    private static string Describe(string label, string? description) => string.IsNullOrEmpty(description) ? label : $"{label}: {description}";
    private static string QuestionId(int index) => "question-" + DecisionRequests.Index(index);
    private static string OptionId(int index) => "option-" + DecisionRequests.Index(index);
    private static void Warn(List<object> warnings, string? feature, string message) => warnings.Add(new { type = "other", message, feature });
}
