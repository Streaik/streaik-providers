using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.OpenAI.Models.Decisions;
using Streaik.Providers.Decisions;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class UpstageProvider
{
    private const double ProbabilityTolerance = 1e-6;

    public async Task<DecisionResponse> DecideAsync(DecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        DecisionRequests.Validate(request);
        if (DecisionRequests.Options(request, Id).Any())
            throw new ProviderRequestException("Upstage SystemOne does not support additional decision options.");

        var questions = new JsonObject();
        foreach (var (id, question) in request.Questions)
        {
            var mapped = new JsonObject { ["type"] = NativeType(question), ["instructions"] = Copy(question.Instructions!.Value) };
            switch (question)
            {
                case DecisionBooleanQuestion boolean:
                    if (boolean.Criteria is not null) mapped["criteria"] = JsonSerializer.SerializeToNode(boolean.Criteria, Json);
                    break;
                case DecisionChoiceQuestion choice when choice.Criteria.Count is >= 2 and <= 26 && choice.Criteria.Keys.All(key => !string.IsNullOrWhiteSpace(key)):
                    mapped["criteria"] = JsonSerializer.SerializeToNode(choice.Criteria, Json);
                    break;
                case DecisionScoreQuestion score when score.Criteria.Count >= 2 && score.Criteria.All(level => level is not null):
                    mapped["criteria"] = JsonSerializer.SerializeToNode(score.Criteria, Json);
                    break;
                default: throw new ProviderRequestException($"Invalid criteria for '{id}': Upstage requires 2–26 named choices or at least two nonnull score levels.");
            }
            questions[id] = mapped;
        }

        var payload = new JsonObject { ["model"] = request.Model, ["state"] = Copy(DecisionRequests.State(request)), ["questions"] = questions };
        using var client = CreateClient();
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, "v1/systemone", payload, Bearer(ApiKey()), headers, Json, ct);
        var model = String(body, "model");
        if (string.IsNullOrWhiteSpace(model)) throw new JsonException("Missing Upstage decision model version.");
        var nativeAnswers = Property(body, "answers");
        if (nativeAnswers.ValueKind != JsonValueKind.Object || nativeAnswers.EnumerateObject().Count() != request.Questions.Count
            || nativeAnswers.EnumerateObject().Select(answer => answer.Name).Distinct(StringComparer.Ordinal).Count() != request.Questions.Count)
            throw new JsonException("Upstage decision answers must cover every question exactly once.");
        var answers = request.Questions.ToDictionary(pair => pair.Key, pair => MapAnswer(Property(nativeAnswers, pair.Key), pair.Value), StringComparer.Ordinal);
        var usage = Property(body, "usage");
        return new()
        {
            Answers = answers,
            Usage = new() { InputTokens = Tokens(usage, "input_tokens"), OutputTokens = Tokens(usage, "output_tokens") },
            ProviderMetadata = new() { [Id] = body },
            Response = new() { ModelId = $"{Id}/{model}", Timestamp = DateTimeOffset.UtcNow, Id = upstream.GetValueOrDefault("x-request-id"),
                Body = body, Headers = NativeHttp.SafeResponseMetadataHeaders(upstream) }
        };
    }

    public async Task<OpenAIDecisionResponse> DecideAsync(OpenAIDecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        DecisionRequests.Validate(request);
        if (request.SafetyIdentifier is not null || request.PromptCacheKey is not null || request.PromptCacheOptions is not null
            || request.AdditionalProperties?.Count > 0)
            throw new ProviderRequestException("Upstage SystemOne does not support these OpenAI decision options.");
        if (request.Input.Items?.OfType<OpenAIDecisionInputMessage>().Any(message =>
                message.Content.Parts?.Any(part => part is OpenAIDecisionInputImage) == true) == true
            || request.Questions.Any(question => question.InstructionContent.Parts?.Any(part => part is OpenAIDecisionInputImage) == true))
            throw new ProviderRequestException("Upstage SystemOne does not support OpenAI decision image parts.");

        var native = new DecisionRequest { Model = request.Model, State = new(JsonSerializer.SerializeToElement(request.Input, Json)) };
        for (var i = 0; i < request.Questions.Count; i++)
        {
            var question = request.Questions[i];
            DecisionQuestion mapped = question switch
            {
                OpenAIDecisionPredicateQuestion => new DecisionBooleanQuestion(),
                OpenAIDecisionChoiceQuestion choice => new DecisionChoiceQuestion
                {
                    Criteria = choice.Choices.Select((option, j) => KeyValuePair.Create(OptionId(j),
                        (DecisionInput?)new DecisionInput(Describe(option.Value.Value.GetRawText(), option.Description))))
                        .ToDictionary(StringComparer.Ordinal)
                },
                OpenAIDecisionScoreQuestion score => new DecisionScoreQuestion
                {
                    Criteria = score.Levels.Select(level => (DecisionInput?)new DecisionInput(Describe(level.Label, level.Description))).ToArray()
                },
                _ => throw new ProviderRequestException("Unsupported Upstage decision question.")
            };
            mapped.Instructions = new(JsonSerializer.SerializeToElement(question.InstructionContent, Json));
            native.Questions[QuestionId(i)] = mapped;
        }

        var result = await DecideAsync(native, headers, ct);
        var raw = result.ProviderMetadata![Id];
        var answers = new List<OpenAIDecisionAnswer>();
        for (var i = 0; i < request.Questions.Count; i++)
        {
            var id = QuestionId(i);
            var question = request.Questions[i];
            var original = raw.GetProperty("answers").GetProperty(id);
            OpenAIDecisionAnswer mapped = (question, result.Answers[id]) switch
            {
                (OpenAIDecisionPredicateQuestion, DecisionBooleanAnswer boolean) => new OpenAIDecisionPredicateAnswer { Probability = boolean.Probability },
                (OpenAIDecisionChoiceQuestion choice, DecisionChoiceAnswer answer) => new OpenAIDecisionChoiceAnswer
                {
                    Choice = choice.Choices[Enumerable.Range(0, choice.Choices.Count).Single(j => OptionId(j) == answer.Choice)].Value,
                    Confidence = Probability(Property(original, "confidence")),
                    Probabilities = choice.Choices.Select((option, j) => new OpenAIDecisionChoiceProbability
                        { Value = option.Value, Probability = answer.Probabilities![OptionId(j)] }).ToArray()
                },
                (OpenAIDecisionScoreQuestion score, DecisionScoreAnswer answer) => new OpenAIDecisionScoreAnswer
                {
                    Score = answer.Score / (score.Levels.Count - 1d), Confidence = Probability(Property(original, "confidence")),
                    Probabilities = score.Levels.Select((level, j) => new OpenAIDecisionScoreProbability
                        { Label = level.Label, Value = j / (score.Levels.Count - 1d), Probability = answer.Probabilities![DecisionRequests.Index(j)] }).ToArray()
                },
                _ => throw new JsonException("Unexpected Upstage decision answer type.")
            };
            mapped.Name = question.Name;
            answers.Add(mapped);
        }
        var input = result.Usage!.InputTokens!.Value;
        var output = result.Usage.OutputTokens!.Value;
        if (input > int.MaxValue - output) throw new JsonException("Invalid Upstage decision token total.");
        return new()
        {
            Model = String(raw, "model")!, Answers = answers, Usage = new() { InputTokens = input, OutputTokens = output, TotalTokens = input + output },
            AdditionalProperties = new()
            {
                ["providerMetadata"] = JsonSerializer.SerializeToElement(result.ProviderMetadata, Json),
                ["warnings"] = JsonSerializer.SerializeToElement(new[] { new { type = "other",
                    message = "Upstage does not report cached, cache-write, or reasoning tokens. Score values use an evenly spaced [0,1] scale. Solar Decide output tokens are reported but not billed." } }, Json)
            }
        };
    }

    private static DecisionAnswer MapAnswer(JsonElement answer, DecisionQuestion question)
    {
        if (String(answer, "type") != NativeType(question)) throw new JsonException("Upstage decision answer type does not match its question.");
        if (question is DecisionBooleanQuestion) return new DecisionBooleanAnswer { Probability = Probability(Property(answer, "noul")) };
        _ = Probability(Property(answer, "confidence"));
        if (question is DecisionChoiceQuestion choice)
        {
            var probabilities = Distribution(Property(answer, "probabilities"), choice.Criteria.Keys);
            var selected = String(answer, "choice");
            if (selected is null || !probabilities.TryGetValue(selected, out var probability) || probability + ProbabilityTolerance < probabilities.Values.Max())
                throw new JsonException("Upstage returned a choice that was not a maximal option.");
            return new DecisionChoiceAnswer { Choice = selected, Probabilities = probabilities };
        }
        if (question is not DecisionScoreQuestion score) throw new JsonException("Unsupported Upstage decision question.");
        var keys = Enumerable.Range(0, score.Criteria.Count).Select(DecisionRequests.Index).ToArray();
        var distribution = Distribution(Property(answer, "probabilities"), keys);
        var legend = Property(answer, "legend");
        if (legend.ValueKind != JsonValueKind.Object || legend.EnumerateObject().Count() != keys.Length
            || keys.Where((key, i) => !JsonElement.DeepEquals(Property(legend, key), score.Criteria[i]!.Value)).Any())
            throw new JsonException("Upstage score legend must match the submitted rubric.");
        var value = Property(answer, "score");
        var expected = keys.Select((key, i) => i * distribution[key]).Sum();
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetDouble(out var number) || !double.IsFinite(number)
            || number < 0 || number > keys.Length - 1 || Math.Abs(number - expected) > ProbabilityTolerance)
            throw new JsonException("Invalid Upstage weighted score.");
        return new DecisionScoreAnswer { Score = number, Probabilities = distribution };
    }

    private static Dictionary<string, double> Distribution(JsonElement json, IEnumerable<string> keys)
    {
        var expected = keys.ToHashSet(StringComparer.Ordinal);
        var values = new Dictionary<string, double>(StringComparer.Ordinal);
        if (json.ValueKind != JsonValueKind.Object) throw new JsonException("Missing Upstage decision probabilities.");
        foreach (var entry in json.EnumerateObject())
            if (!expected.Contains(entry.Name) || !values.TryAdd(entry.Name, Probability(entry.Value)))
                throw new JsonException("Unexpected or duplicate Upstage decision probability key.");
        if (values.Count != expected.Count || Math.Abs(values.Values.Sum() - 1) > ProbabilityTolerance)
            throw new JsonException("Incomplete or unnormalized Upstage decision probabilities.");
        return values;
    }

    private static double Probability(JsonElement value) => value.ValueKind == JsonValueKind.Number && value.TryGetDouble(out var number)
        && double.IsFinite(number) && number is >= 0 and <= 1 ? number : throw new JsonException("Invalid Upstage decision probability.");
    private static int Tokens(JsonElement usage, string name) => Property(usage, name) is { ValueKind: JsonValueKind.Number } value
        && value.TryGetInt32(out var count) && count >= 0 ? count : throw new JsonException("Invalid Upstage decision usage.");
    private static string NativeType(DecisionQuestion question) => question is DecisionBooleanQuestion ? "noul" : question.Type;
    private static string QuestionId(int i) => "question-" + DecisionRequests.Index(i);
    private static string OptionId(int i) => "option-" + DecisionRequests.Index(i);
    private static string Describe(string label, string? description) => string.IsNullOrEmpty(description) ? label : $"{label}: {description}";
}
