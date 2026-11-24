using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.OpenAI.Models.Decisions;
using Streaik.Providers.Decisions;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class DigitalOceanProvider
{
    public async Task<DecisionResponse> DecideAsync(DecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        DecisionRequests.Validate(request);
        if (DecisionRequests.Options(request, Id).Any())
            throw new ProviderRequestException("DigitalOcean System One does not support decision provider options.");

        var state = DecisionRequests.State(request, allowImages: false);
        if (request.StateFormat == "sdk") state = JsonSerializer.SerializeToElement(state[0].GetProperty("content")
            .EnumerateArray().Select(part => part.GetProperty("text").GetString()).ToArray(), Json);
        if (state.ValueKind is not (JsonValueKind.String or JsonValueKind.Array or JsonValueKind.Object)
            || state.ValueKind == JsonValueKind.Array && state.EnumerateArray().Any(value => value.ValueKind != JsonValueKind.String))
            throw new ProviderRequestException("DigitalOcean System One state must be text, a text array, or a JSON object.");

        var questions = new JsonObject();
        foreach (var (id, question) in request.Questions)
        {
            if (question.Instructions!.Value is not { ValueKind: JsonValueKind.String } instructions || string.IsNullOrWhiteSpace(instructions.GetString()))
                throw new ProviderRequestException($"DigitalOcean question '{id}' requires text instructions.");
            var mapped = new JsonObject { ["type"] = NativeType(question), ["instructions"] = instructions.GetString() };
            switch (question)
            {
                case DecisionBooleanQuestion { Criteria: null }: break;
                case DecisionChoiceQuestion choice when choice.Criteria.All(option => !string.IsNullOrWhiteSpace(option.Key)
                    && option.Value?.Value.ValueKind == JsonValueKind.String):
                    mapped["criteria"] = JsonSerializer.SerializeToNode(choice.Criteria, Json);
                    break;
                case DecisionScoreQuestion score when score.Criteria.All(level => level?.Value is { } value
                    && (value.ValueKind == JsonValueKind.String || String(value, "label") is not null)):
                    mapped["criteria"] = JsonSerializer.SerializeToNode(score.Criteria, Json);
                    break;
                default: throw new ProviderRequestException($"Unsupported DigitalOcean criteria for '{id}'.");
            }
            questions[id] = mapped;
        }

        using var client = CreateClient();
        var payload = new JsonObject { ["model"] = request.Model, ["state"] = Copy(state), ["questions"] = questions };
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, "v1/systemone", payload, Bearer(ApiKey()), headers, Json, ct);
        var model = DecisionResults.RequiredModel(body);
        var rawAnswers = Property(body, "answers");
        if (rawAnswers.ValueKind != JsonValueKind.Object || rawAnswers.EnumerateObject().Count() != request.Questions.Count
            || rawAnswers.EnumerateObject().Select(answer => answer.Name).Distinct(StringComparer.Ordinal).Count() != request.Questions.Count)
            throw new JsonException("DigitalOcean decision answers must cover every question exactly once.");
        var answers = request.Questions.ToDictionary(pair => pair.Key, pair => ReadAnswer(Property(rawAnswers, pair.Key), pair.Value), StringComparer.Ordinal);
        var (input, output) = DecisionResults.Usage(body);
        return DecisionResults.Sdk(Id, model, body, NativeHttp.SafeResponseMetadataHeaders(upstream), answers, input, output);
    }

    public async Task<OpenAIDecisionResponse> DecideAsync(OpenAIDecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        DecisionRequests.Validate(request);
        if (request.SafetyIdentifier is not null || request.PromptCacheKey is not null || request.PromptCacheOptions is not null
            || request.AdditionalProperties?.Count > 0)
            throw new ProviderRequestException("DigitalOcean System One does not support those OpenAI decision options.");
        var native = new DecisionRequest { Model = request.Model, State = new(OpenAIState(request.Input)) };
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
                _ => throw new ProviderRequestException("Unsupported DigitalOcean decision question.")
            };
            mapped.Instructions = new(question.InstructionContent.Text ?? throw new ProviderRequestException("DigitalOcean requires text instructions."));
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
                    Confidence = DecisionResults.Probability(Property(original, "confidence")),
                    Probabilities = choice.Choices.Select((option, j) => new OpenAIDecisionChoiceProbability
                        { Value = option.Value, Probability = answer.Probabilities![OptionId(j)] }).ToArray()
                },
                (OpenAIDecisionScoreQuestion score, DecisionScoreAnswer answer) => new OpenAIDecisionScoreAnswer
                {
                    Score = answer.Score, Confidence = DecisionResults.Probability(Property(original, "confidence")),
                    Probabilities = score.Levels.Select((level, j) => new OpenAIDecisionScoreProbability
                        { Label = level.Label, Value = j, Probability = answer.Probabilities![DecisionRequests.Index(j)] }).ToArray()
                },
                _ => throw new JsonException("Unexpected DigitalOcean decision answer type.")
            };
            mapped.Name = question.Name;
            answers.Add(mapped);
        }
        var (input, output) = DecisionResults.Usage(raw);
        if (input > int.MaxValue - output) throw new JsonException("Invalid DigitalOcean token total.");
        return new()
        {
            Model = DecisionResults.RequiredModel(raw), Answers = answers,
            Usage = new() { InputTokens = input, OutputTokens = output, TotalTokens = input + output },
            AdditionalProperties = new()
            {
                ["providerMetadata"] = JsonSerializer.SerializeToElement(result.ProviderMetadata, Json),
                ["warnings"] = JsonSerializer.SerializeToElement(new[] { new { type = "other",
                    message = "DigitalOcean uses zero-based level positions for score values and does not report cached or reasoning tokens." } }, Json)
            }
        };
    }

    private static JsonElement OpenAIState(OpenAIDecisionInput input)
    {
        if (input.Text is { } text) return JsonSerializer.SerializeToElement(text, Json);
        var parts = new List<string>();
        foreach (var item in input.Items ?? [])
        {
            if (item is not OpenAIDecisionInputMessage message)
                throw new ProviderRequestException("DigitalOcean decision input does not support tool items or cache breakpoints.");
            if (message.Content.Text is { } content) parts.Add(content);
            else foreach (var part in message.Content.Parts ?? [])
                parts.Add(part switch
                {
                    OpenAIDecisionInputText value => value.Text,
                    OpenAIDecisionOutputText value => value.Text,
                    OpenAIDecisionInputJson value => value.Data.GetRawText(),
                    _ => throw new ProviderRequestException("DigitalOcean decision input only supports text and JSON evidence.")
                });
        }
        return JsonSerializer.SerializeToElement(parts, Json);
    }

    private static DecisionAnswer ReadAnswer(JsonElement answer, DecisionQuestion question)
    {
        if (String(answer, "type") != NativeType(question)) throw new JsonException("Decision answer type does not match its question.");
        if (question is DecisionBooleanQuestion) return new DecisionBooleanAnswer { Probability = DecisionResults.Probability(Property(answer, "noul")) };
        _ = DecisionResults.Probability(Property(answer, "confidence"));
        var probabilities = Property(answer, "probabilities");
        if (probabilities.ValueKind != JsonValueKind.Object) throw new JsonException("Missing DigitalOcean decision probabilities.");
        var values = probabilities.EnumerateObject().Select(entry => (entry.Name, DecisionResults.Probability(entry.Value)));
        if (question is DecisionChoiceQuestion choice)
        {
            var distribution = DecisionResults.Distribution(values, choice.Criteria.Keys);
            if (String(answer, "choice") is not { } selected || !distribution.TryGetValue(selected, out var probability)
                || probability + 1e-6 < distribution.Values.Max()) throw new JsonException("DigitalOcean selected a non-maximal choice.");
            return new DecisionChoiceAnswer { Choice = selected, Probabilities = distribution };
        }
        if (question is not DecisionScoreQuestion score) throw new JsonException("Unexpected DigitalOcean decision question type.");
        var keys = Enumerable.Range(0, score.Criteria.Count).Select(DecisionRequests.Index).ToArray();
        var scores = DecisionResults.Distribution(values, keys);
        var legend = Property(answer, "legend");
        if (legend.ValueKind != JsonValueKind.Object || legend.EnumerateObject().Count() != keys.Length || keys.Any(key => String(legend, key) is null))
            throw new JsonException("Invalid DigitalOcean score legend.");
        var weighted = DecisionResults.Number(Property(answer, "score"));
        if (weighted < 0 || weighted > keys.Length - 1 || Math.Abs(weighted - keys.Select((key, i) => i * scores[key]).Sum()) > 1e-6)
            throw new JsonException("Invalid DigitalOcean weighted score.");
        return new DecisionScoreAnswer { Score = weighted, Probabilities = scores };
    }

    private static string NativeType(DecisionQuestion question) => question is DecisionBooleanQuestion ? "noul" : question.Type;
    private static string QuestionId(int index) => "question-" + DecisionRequests.Index(index);
    private static string OptionId(int index) => "option-" + DecisionRequests.Index(index);
    private static string Describe(string label, string? description) => string.IsNullOrEmpty(description) ? label : $"{label}: {description}";
}
