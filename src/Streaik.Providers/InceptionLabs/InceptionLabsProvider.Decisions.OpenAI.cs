using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.OpenAI.Models.Decisions;
using Streaik.Providers.Decisions;
using Streaik.Vercel.Models;

namespace Streaik.Providers;

internal sealed partial class InceptionLabsProvider
{
    public async Task<OpenAIDecisionResponse> DecideAsync(
        OpenAIDecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        DecisionRequests.Validate(request);
        if (request.SafetyIdentifier is not null || request.PromptCacheKey is not null || request.PromptCacheOptions is not null || request.AdditionalProperties?.Count > 0)
            throw new ProviderRequestException("Inception Labs Decisions does not support safety_identifier or additional OpenAI fields, including streaming and extraction fields.");
        var typed = new DecisionRequest
        {
            Model = request.Model,
            State = request.Input.Text is not null
                ? new DecisionInput(request.Input.Text)
                : new DecisionInput(JsonSerializer.SerializeToElement(
                    MapInceptionDecisionMessages(JsonSerializer.SerializeToNode(request.Input) ?? throw new ArgumentException("Decision input is required."))))
        };
        for (var index = 0; index < request.Questions.Count; index++)
        {
            var question = request.Questions[index];
            typed.Questions[InceptionQuestionId(index)] = question switch
            {
                OpenAIDecisionPredicateQuestion => new DecisionBooleanQuestion { Instructions = Instructions(question) },
                OpenAIDecisionChoiceQuestion choice => new DecisionChoiceQuestion
                {
                    Instructions = Instructions(question),
                    Criteria = choice.Choices.Select((option, i) => KeyValuePair.Create(InceptionOptionId(i),
                        (DecisionInput?)new DecisionInput(JsonSerializer.SerializeToElement(new
                        {
                            value = option.Value, description = option.Description
                        })))).ToDictionary(item => item.Key, item => item.Value, StringComparer.Ordinal)
                },
                OpenAIDecisionScoreQuestion score => new DecisionScoreQuestion
                {
                    Instructions = Instructions(question),
                    Criteria = score.Levels.Select(level => (DecisionInput?)new DecisionInput(
                        JsonSerializer.SerializeToElement(level))).ToArray()
                },
                _ => throw new ArgumentException("Unsupported decision question type.", nameof(request))
            };
        }

        var result = await DecideAsync(typed, headers, ct);
        var raw = result.ProviderMetadata![Id];
        var answers = new List<OpenAIDecisionAnswer>();
        for (var index = 0; index < request.Questions.Count; index++)
        {
            var id = InceptionQuestionId(index);
            var question = request.Questions[index];
            var native = raw.GetProperty("answers").GetProperty(id);
            OpenAIDecisionAnswer answer = (question, result.Answers[id]) switch
            {
                (OpenAIDecisionPredicateQuestion, DecisionBooleanAnswer boolean) =>
                    new OpenAIDecisionPredicateAnswer { Probability = boolean.Probability },
                (OpenAIDecisionChoiceQuestion choice, DecisionChoiceAnswer selected) => new OpenAIDecisionChoiceAnswer
                {
                    Choice = choice.Choices[int.Parse(selected.Choice["option-".Length..], CultureInfo.InvariantCulture)].Value,
                    Confidence = native.GetProperty("confidence").GetDouble(),
                    Probabilities = choice.Choices.Select((option, i) => new OpenAIDecisionChoiceProbability
                    {
                        Value = option.Value, Probability = selected.Probabilities![InceptionOptionId(i)]
                    }).ToArray()
                },
                (OpenAIDecisionScoreQuestion score, DecisionScoreAnswer scored) => new OpenAIDecisionScoreAnswer
                {
                    Score = scored.Score,
                    Confidence = native.GetProperty("confidence").GetDouble(),
                    Probabilities = score.Levels.Select((level, i) => new OpenAIDecisionScoreProbability
                    {
                        Label = level.Label, Value = i,
                        Probability = scored.Probabilities![InceptionDecisionIndex(i)]
                    }).ToArray()
                },
                _ => throw new InvalidOperationException("Unexpected Inception decision answer type.")
            };
            answer.Name = question.Name;
            answers.Add(answer);
        }
        var input = result.Usage!.InputTokens!.Value;
        var output = result.Usage.OutputTokens!.Value;
        return new()
        {
            Model = raw.GetProperty("model").GetString()!, Answers = answers,
            Usage = new() { InputTokens = input, OutputTokens = output, TotalTokens = checked(input + output) },
            AdditionalProperties = new()
            {
                ["providerMetadata"] = JsonSerializer.SerializeToElement(result.ProviderMetadata),
                ["warnings"] = JsonSerializer.SerializeToElement(new[] { new
                {
                    type = "other",
                    message = "Inception Labs does not report cached, cache-write, or reasoning tokens; OpenAI-compatible detail counters default to zero. Scores use zero-based level positions."
                } })
            }
        };
    }

    private static string InceptionQuestionId(int index) => "question-" + InceptionDecisionIndex(index);
    private static string InceptionOptionId(int index) => "option-" + InceptionDecisionIndex(index);

    private static DecisionInput Instructions(OpenAIDecisionQuestion question)
    {
        if (question.InstructionContent.Text is { } text) return new(text);
        var parts = JsonSerializer.SerializeToNode(question.InstructionContent)!.AsArray();
        MapInceptionDecisionParts(parts);
        return new(JsonSerializer.SerializeToElement(parts));
    }
}
