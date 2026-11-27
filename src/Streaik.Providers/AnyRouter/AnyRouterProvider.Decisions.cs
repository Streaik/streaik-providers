using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.OpenAI.Models.Decisions;
using Streaik.Providers.Decisions;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class AnyRouterProvider
{
    public async Task<DecisionResponse> DecideAsync(DecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        DecisionRequests.Validate(request);
        if (DecisionRequests.Options(request, Id).Any())
            throw new ProviderRequestException("AnyRouter Decisions has no documented provider options.");
        var (body, upstream) = await SendDecisionAsync(request.Model, DecisionRequests.State(request), request.Questions, headers, ct);
        var answers = ReadAnswers(body, request.Questions);
        var (input, output) = ReadUsage(body);
        return new()
        {
            Answers = answers, Usage = new() { InputTokens = input, OutputTokens = output },
            ProviderMetadata = NativeMediaOptions.Metadata(Id, body), Warnings = [],
            Response = new() { ModelId = DecisionResults.RequiredModel(body), Timestamp = DateTimeOffset.UtcNow,
                Id = String(body, "id") ?? upstream.GetValueOrDefault("x-request-id"), Body = body,
                Headers = NativeHttp.SafeResponseMetadataHeaders(upstream) }
        };
    }

    public async Task<OpenAIDecisionResponse> DecideAsync(OpenAIDecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        DecisionRequests.Validate(request);
        if (request.SafetyIdentifier is not null || request.AdditionalProperties?.Count > 0)
            throw new ProviderRequestException("AnyRouter Decisions does not support safety_identifier or extra fields.");
        var questions = new Dictionary<string, DecisionQuestion>(StringComparer.Ordinal);
        for (var i = 0; i < request.Questions.Count; i++)
        {
            var source = request.Questions[i];
            questions[QuestionId(i)] = source switch
            {
                OpenAIDecisionPredicateQuestion => new DecisionBooleanQuestion { Instructions = source.Instructions },
                OpenAIDecisionChoiceQuestion choice => new DecisionChoiceQuestion { Instructions = source.Instructions,
                    Criteria = choice.Choices.Select((c, index) => (c, index)).ToDictionary(p => OptionId(p.index),
                        p => p.c.Description is { } description ? new DecisionInput(description) : null) },
                OpenAIDecisionScoreQuestion score => new DecisionScoreQuestion { Instructions = source.Instructions,
                    Criteria = score.Levels.Select(l => (DecisionInput?)new DecisionInput(l.Description ?? l.Label)).ToArray() },
                _ => throw new ProviderRequestException("Unsupported AnyRouter decision question.")
            };
        }
        var state = request.Input.Text is { } textInput ? JsonSerializer.SerializeToElement(textInput) : JsonSerializer.SerializeToElement(request.Input.Messages, Json);
        var (body, _) = await SendDecisionAsync(request.Model, state, questions, headers, ct);
        var typed = ReadAnswers(body, questions);
        var answers = new List<OpenAIDecisionAnswer>();
        for (var i = 0; i < request.Questions.Count; i++)
        {
            var source = request.Questions[i];
            var id = QuestionId(i);
            var native = body.GetProperty("answers").GetProperty(id);
            OpenAIDecisionAnswer answer = (source, typed[id]) switch
            {
                (OpenAIDecisionPredicateQuestion, DecisionBooleanAnswer b) => new OpenAIDecisionPredicateAnswer { Probability = b.Probability },
                (OpenAIDecisionChoiceQuestion choice, DecisionChoiceAnswer a) => new OpenAIDecisionChoiceAnswer
                {
                    Choice = choice.Choices[Enumerable.Range(0, choice.Choices.Count).Single(i => OptionId(i) == a.Choice)].Value,
                    Confidence = DecisionResults.Probability(Property(native, "confidence")),
                    Probabilities = choice.Choices.Select((c, index) => new OpenAIDecisionChoiceProbability
                        { Value = c.Value, Probability = a.Probabilities![OptionId(index)] }).ToArray()
                },
                (OpenAIDecisionScoreQuestion score, DecisionScoreAnswer a) => new OpenAIDecisionScoreAnswer
                {
                    Score = a.Score, Confidence = DecisionResults.Probability(Property(native, "confidence")),
                    Probabilities = score.Levels.Select((l, index) => new OpenAIDecisionScoreProbability
                        { Label = l.Label, Value = index, Probability = a.Probabilities![DecisionRequests.Index(index)] }).ToArray()
                },
                _ => throw new JsonException("Unexpected AnyRouter answer type.")
            };
            answer.Name = source.Name;
            answers.Add(answer);
        }
        var (input, output) = ReadUsage(body);
        if ((long)(input ?? 0) + (output ?? 0) > int.MaxValue) throw new JsonException("Invalid AnyRouter decision token total.");
        return new()
        {
            Model = DecisionResults.RequiredModel(body), Answers = answers,
            Usage = new() { InputTokens = input ?? 0, OutputTokens = output ?? 0, TotalTokens = (input ?? 0) + (output ?? 0) },
            AdditionalProperties = new() { ["providerMetadata"] = JsonSerializer.SerializeToElement(NativeMediaOptions.Metadata(Id, body), Json) }
        };
    }

    private async Task<(JsonElement Body, Dictionary<string, string?> Headers)> SendDecisionAsync(string model, JsonElement state,
        Dictionary<string, DecisionQuestion> questions, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        return await NativeHttp.PostJsonAsync(client, "decisions",
            new { model, state, questions = questions.ToDictionary(p => p.Key, p => NativeQuestion(p.Value), StringComparer.Ordinal) },
            Bearer(ApiKey()), headers, Json, ct);
    }

    private static JsonObject NativeQuestion(DecisionQuestion question)
    {
        var node = new JsonObject { ["type"] = question is DecisionBooleanQuestion ? "noul" : question.Type,
            ["instructions"] = Render(question.Instructions ?? throw new ProviderRequestException("AnyRouter requires instructions.")) };
        switch (question)
        {
            case DecisionBooleanQuestion { Criteria: { } c }:
                var criteria = new JsonObject();
                if (c.True is { } yes) criteria["true"] = Render(yes);
                if (c.False is { } no) criteria["false"] = Render(no);
                if (criteria.Count > 0) node["criteria"] = criteria;
                break;
            case DecisionBooleanQuestion: break;
            case DecisionChoiceQuestion choice:
                node["criteria"] = new JsonObject(choice.Criteria.Select(p => KeyValuePair.Create<string, JsonNode?>(p.Key,
                    p.Value is null ? null : JsonValue.Create(Render(p.Value)))));
                break;
            case DecisionScoreQuestion score:
                if (score.Criteria.Count is < 2 or > 10 || score.Criteria.Any(c => c is null))
                    throw new ProviderRequestException("AnyRouter score criteria require 2 to 10 ordered descriptions.");
                node["criteria"] = new JsonArray(score.Criteria.Select(c => (JsonNode?)JsonValue.Create(Render(c!))).ToArray());
                break;
            default: throw new ProviderRequestException("Unsupported AnyRouter decision question.");
        }
        return node;
    }

    private static Dictionary<string, DecisionAnswer> ReadAnswers(JsonElement body, Dictionary<string, DecisionQuestion> questions)
    {
        var answers = Property(body, "answers");
        if (answers.ValueKind != JsonValueKind.Object || answers.EnumerateObject().Count() != questions.Count)
            throw new JsonException("AnyRouter answers must match the submitted question IDs.");
        return questions.ToDictionary(p => p.Key, p => ReadAnswer(Property(answers, p.Key), p.Value), StringComparer.Ordinal);
    }

    private static DecisionAnswer ReadAnswer(JsonElement answer, DecisionQuestion question)
    {
        if (String(answer, "type") != (question is DecisionBooleanQuestion ? "noul" : question.Type))
            throw new JsonException("AnyRouter answer type does not match the question.");
        if (question is DecisionBooleanQuestion) return new DecisionBooleanAnswer { Probability = DecisionResults.Probability(Property(answer, "noul")) };
        _ = DecisionResults.Probability(Property(answer, "confidence"));
        var raw = Property(answer, "probabilities");
        if (raw.ValueKind != JsonValueKind.Object) throw new JsonException("Missing AnyRouter probabilities.");
        var entries = raw.EnumerateObject().Select(p => (p.Name, DecisionResults.Probability(p.Value)));
        if (question is DecisionChoiceQuestion choice)
        {
            var values = DecisionResults.Distribution(entries, choice.Criteria.Keys);
            if (String(answer, "choice") is not { } selected || !values.TryGetValue(selected, out var probability)
                || probability + 1e-6 < values.Values.Max()) throw new JsonException("Invalid AnyRouter choice.");
            return new DecisionChoiceAnswer { Choice = selected, Probabilities = values };
        }
        if (question is not DecisionScoreQuestion score) throw new JsonException("Unsupported AnyRouter answer.");
        var keys = Enumerable.Range(0, score.Criteria.Count).Select(DecisionRequests.Index).ToArray();
        var distribution = DecisionResults.Distribution(entries, keys);
        var legend = Property(answer, "legend");
        if (legend.ValueKind != JsonValueKind.Object || legend.EnumerateObject().Count() != keys.Length
            || keys.Any(k => Property(legend, k).ValueKind is not (JsonValueKind.String or JsonValueKind.Null)))
            throw new JsonException("Invalid AnyRouter score legend.");
        var value = DecisionResults.Number(Property(answer, "score"));
        if (value < 0 || value > keys.Length - 1 || Math.Abs(value - keys.Select((k, i) => i * distribution[k]).Sum()) > 0.001)
            throw new JsonException("Invalid AnyRouter weighted score.");
        return new DecisionScoreAnswer { Score = value, Probabilities = distribution };
    }

    private static (int? Input, int? Output) ReadUsage(JsonElement body)
    {
        var usage = Property(body, "usage");
        int? TokenCount(string name) => Property(usage, name).ValueKind == JsonValueKind.Undefined ? null : DecisionResults.Integer(usage, name);
        var input = TokenCount("input_tokens");
        var output = TokenCount("output_tokens");
        if (input is null && output is null) throw new JsonException("AnyRouter decision response requires usage.");
        return (input, output);
    }

    private static string Render(DecisionInput input) => input.Value.ValueKind == JsonValueKind.String ? input.Value.GetString()! : input.Value.GetRawText();
    private static string QuestionId(int i) => "question-" + DecisionRequests.Index(i);
    private static string OptionId(int i) => "option-" + DecisionRequests.Index(i);
}
