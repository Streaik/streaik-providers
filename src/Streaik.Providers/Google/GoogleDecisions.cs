using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.OpenAI.Models.Decisions;
using Streaik.Providers.Decisions;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class GoogleGeminiProvider : IDecisionsProvider
{
    private sealed record GoogleQuestion(string Id, string? Name, JsonObject Payload, OpenAIDecisionChoiceValue[]? Choices, string[]? Labels);

    public async Task<OpenAIDecisionResponse> DecideAsync(OpenAIDecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => (await SendDecisionAsync(request, null, headers, ct)).Response;

    public async Task<DecisionResponse> DecideAsync(DecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var native = DecisionRequests.ToOpenAI(request, "google", out var rawInput);
        var (result, body, responseHeaders) = await SendDecisionAsync(native, rawInput, headers, ct);
        var answers = request.Questions.Select((question, i) => (question.Key, Answer: OpenAICompatibleDecisions.MapAnswer(question.Value, result.Answers[i])))
            .ToDictionary(item => item.Key, item => item.Answer, StringComparer.Ordinal);
        return DecisionResults.Sdk("google", result.Model, body, responseHeaders, answers, result.Usage.InputTokens, result.Usage.OutputTokens);
    }

    private async Task<(OpenAIDecisionResponse Response, JsonElement Body, Dictionary<string, string?> Headers)> SendDecisionAsync(
        OpenAIDecisionRequest request, JsonElement? rawInput, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        DecisionRequests.Validate(request);
        if (request.SafetyIdentifier is not null || request.PromptCacheKey is not null || request.PromptCacheOptions is not null)
            throw new ProviderRequestException("Google Decisions does not support prompt-cache controls or safety_identifier.", 422);
        var wire = JsonSerializer.SerializeToElement(request, DecisionRequests.Json);
        var questions = request.Questions.Select((q, index) => MapGoogleQuestion(q, index,
            Property(Array(Property(wire, "questions")).ElementAt(index), "instructions"))).ToArray();
        var model = NormalizeModel(request.Model);
        var payload = new JsonObject();
        foreach (var (name, value) in request.AdditionalProperties ?? [])
        {
            if (name is not ("labels" or "safety_settings")) throw new ProviderRequestException($"Unsupported Google Decisions option '{name}'.", 422);
            payload[name] = Copy(value);
        }
        payload["model"] = model;
        payload["input"] = GoogleDecisionInput(rawInput ?? Property(wire, "input"));
        payload["questions"] = new JsonObject(questions.Select(q => new KeyValuePair<string, JsonNode?>(q.Id, q.Payload)));
        using var client = CreateClient();
        var (body, responseHeaders) = await NativeHttp.PostJsonAsync(client, "v1beta/decisions", payload, GoogleAuth(ApiKey()), headers, Json, ct);
        var answers = Property(body, "answers");
        if (answers.ValueKind != JsonValueKind.Object || answers.EnumerateObject().Count() != questions.Length)
            throw new JsonException("Google did not answer each decision question.");
        var usage = Property(body, "usage");
        var input = DecisionResults.Integer(usage, "total_input_tokens");
        var output = DecisionResults.Integer(usage, "total_output_tokens");
        var total = DecisionResults.Integer(usage, "total_tokens");
        if (total != input || output != 0) throw new JsonException("Invalid Google Decisions usage.");
        return (new OpenAIDecisionResponse { Model = String(body, "model") ?? model,
            Answers = questions.Select(q => GoogleDecisionAnswer(q, Property(answers, q.Id))).ToArray(),
            Usage = new() { InputTokens = input, OutputTokens = output, TotalTokens = total } }, body, responseHeaders);
    }

    private static GoogleQuestion MapGoogleQuestion(OpenAIDecisionQuestion question, int index, JsonElement instructions)
    {
        var id = "q" + index.ToString("D4", CultureInfo.InvariantCulture);
        var text = instructions.ValueKind == JsonValueKind.String ? instructions.GetString() : string.Concat(Array(instructions).Select(DecisionText));
        if (string.IsNullOrWhiteSpace(text)) throw new ProviderRequestException("Google decision questions require text instructions.", 422);
        var payload = new JsonObject { ["instructions"] = text };
        switch (question)
        {
            case OpenAIDecisionPredicateQuestion:
                payload["type"] = "noul";
                return new(id, question.Name, payload, null, null);
            case OpenAIDecisionChoiceQuestion choice:
                payload["type"] = "choice";
                payload["criteria"] = new JsonObject(choice.Choices.Select((option, i) => new KeyValuePair<string, JsonNode?>(
                    OptionId(i), JsonValue.Create(option.Value.Value.GetRawText() + (string.IsNullOrEmpty(option.Description) ? "" : ": " + option.Description)))));
                return new(id, question.Name, payload, choice.Choices.Select(c => c.Value).ToArray(), null);
            case OpenAIDecisionScoreQuestion score:
                payload["type"] = "score";
                payload["criteria"] = new JsonArray(score.Levels.Select(level => (JsonNode?)JsonValue.Create(level.Label +
                    (string.IsNullOrEmpty(level.Description) ? "" : ": " + level.Description))).ToArray());
                return new(id, question.Name, payload, null, score.Levels.Select(x => x.Label).ToArray());
            default: throw new ProviderRequestException("Unsupported Google decision question.", 422);
        }
    }

    private static string OptionId(int index) => "o" + index.ToString("D3", CultureInfo.InvariantCulture);
    private static OpenAIDecisionAnswer GoogleDecisionAnswer(GoogleQuestion question, JsonElement answer)
    {
        var confidence = DecisionResults.Probability(Property(answer, "confidence"));
        var values = Property(answer, "probabilities");
        if (question.Choices is { } choices)
        {
            var keys = Enumerable.Range(0, choices.Length).Select(OptionId).ToArray();
            var probabilities = GoogleDistribution(values, keys);
            var selected = System.Array.IndexOf(keys, String(answer, "choice"));
            if (selected < 0 || probabilities[keys[selected]] + 0.01 < probabilities.Values.Max()) throw new JsonException("Invalid Google decision choice.");
            return new OpenAIDecisionChoiceAnswer { Name = question.Name, Choice = choices[selected], Confidence = confidence,
                Probabilities = keys.Select((key, i) => new OpenAIDecisionChoiceProbability { Value = choices[i], Probability = probabilities[key] }).ToArray() };
        }
        if (question.Labels is { } labels)
        {
            var keys = Enumerable.Range(0, labels.Length).Select(DecisionRequests.Index).ToArray();
            var probabilities = GoogleDistribution(values, keys);
            var score = DecisionResults.Number(Property(answer, "score"));
            if (score < 0 || score > labels.Length - 1) throw new JsonException("Invalid Google decision score.");
            return new OpenAIDecisionScoreAnswer { Name = question.Name, Score = score, Confidence = confidence,
                Probabilities = keys.Select((key, i) => new OpenAIDecisionScoreProbability { Value = i, Label = labels[i], Probability = probabilities[key] }).ToArray() };
        }
        return new OpenAIDecisionPredicateAnswer { Name = question.Name, Probability = DecisionResults.Probability(Property(answer, "noul")) };
    }

    private static Dictionary<string, double> GoogleDistribution(JsonElement values, string[] keys)
    {
        if (values.ValueKind != JsonValueKind.Object) throw new JsonException("Missing Google decision probabilities.");
        return DecisionResults.Distribution(values.EnumerateObject().Select(x => (x.Name, DecisionResults.Probability(x.Value))), keys);
    }

    private static JsonNode GoogleDecisionInput(JsonElement input)
    {
        if (input.ValueKind == JsonValueKind.String) return JsonValue.Create(input.GetString())!;
        var parts = new JsonArray();
        foreach (var message in Array(input))
        {
            var content = Property(message, "content");
            if (content.ValueKind == JsonValueKind.String) { parts.Add(TextPart(content.GetString()!)); continue; }
            foreach (var part in Array(content.ValueKind == JsonValueKind.Array ? content : input))
            {
                if (String(part, "type") == "input_image") parts.Add(GoogleImagePart(String(part, "image_url"), null));
                else parts.Add(TextPart(DecisionText(part)));
            }
            if (content.ValueKind != JsonValueKind.Array) break;
        }
        if (parts.Count == 0) throw new ProviderRequestException("Missing Google decision input.");
        return parts;
    }

    private static string DecisionText(JsonElement part) => String(part, "text")
        ?? (Property(part, "data") is { ValueKind: JsonValueKind.Array or JsonValueKind.Object } json ? json.GetRawText()
            : throw new ProviderRequestException("Google Decisions accepts text, JSON, and image evidence on compatible routes.", 422));
}
