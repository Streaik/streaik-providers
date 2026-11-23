using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.OpenAI.Models.Decisions;
using Streaik.Vercel.Models;

namespace Streaik.Providers.Decisions;

public static class OpenAICompatibleDecisions
{
    internal static async Task<OpenAIDecisionResponse> DecideAsync(HttpClient client, string path, string key,
        OpenAIDecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => (await SendAsync(client, path, key, request, null, headers, ct)).Result;

    internal static async Task<DecisionResponse> DecideAsync(HttpClient client, string path, string key, string provider,
        DecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var native = DecisionRequests.ToOpenAI(request, provider, out var rawInput);
        var (result, body, upstreamHeaders) = await SendAsync(client, path, key, native, rawInput, headers, ct);
        var answers = request.Questions.Zip(result.Answers).ToDictionary(x => x.First.Key,
            x => MapAnswer(x.First.Value, x.Second), StringComparer.Ordinal);
        return DecisionResults.Sdk(provider, result.Model, body, upstreamHeaders, answers, result.Usage.InputTokens, result.Usage.OutputTokens);
    }

    private static async Task<(OpenAIDecisionResponse Result, JsonElement Body, Dictionary<string, string?> Headers)> SendAsync(
        HttpClient client, string path, string key, OpenAIDecisionRequest request, JsonElement? rawInput,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        DecisionRequests.Validate(request);
        JsonObject payload;
        try { payload = JsonSerializer.SerializeToNode(request, DecisionRequests.Json)!.AsObject(); }
        catch (JsonException) { throw new ProviderRequestException("Invalid OpenAI Decisions input."); }
        if (rawInput is { } input) payload["input"] = JsonNode.Parse(input.GetRawText());
        var (body, upstreamHeaders) = await NativeHttp.PostJsonAsync(client, path, payload, h => h.Authorization = new("Bearer", key), headers, DecisionRequests.Json, ct);

        DecisionResults.RequiredModel(body);
        DecisionResults.Usage(body);
        var usage = DecisionResults.Property(body, "usage");
        DecisionResults.Integer(usage, "total_tokens");
        DecisionResults.ValidateDetails(usage, "input_tokens_details", "cached_tokens", "cache_write_tokens");
        DecisionResults.ValidateDetails(usage, "output_tokens_details", "reasoning_tokens");
        var rawAnswers = DecisionResults.Property(body, "answers");
        if (rawAnswers.ValueKind != JsonValueKind.Array || rawAnswers.GetArrayLength() != request.Questions.Count)
            throw new JsonException("Decisions requires one answer per question.");
        foreach (var (raw, question) in rawAnswers.EnumerateArray().Zip(request.Questions))
        {
            var type = DecisionResults.String(raw, "type");
            var name = DecisionResults.Property(raw, "name");
            if (name.ValueKind is not (JsonValueKind.String or JsonValueKind.Null or JsonValueKind.Undefined)
                || (name.ValueKind == JsonValueKind.String ? name.GetString() : null) != question.Name
                || type != "refusal" && type != question.Type)
                throw new JsonException("Decision answers must match the question names and types in order.");
            switch (type)
            {
                case "predicate": DecisionResults.Probability(DecisionResults.Property(raw, "probability")); break;
                case "score": DecisionResults.Number(DecisionResults.Property(raw, "score")); goto case "choice";
                case "choice":
                    DecisionResults.Probability(DecisionResults.Property(raw, "confidence"));
                    if (type == "choice" && !DecisionRequests.IsChoice(DecisionResults.Property(raw, "choice"))) throw new JsonException("Invalid selected choice.");
                    var probabilities = DecisionResults.Property(raw, "probabilities");
                    if (probabilities.ValueKind != JsonValueKind.Array) throw new JsonException("Missing decision probabilities.");
                    foreach (var item in probabilities.EnumerateArray())
                    {
                        DecisionResults.Probability(DecisionResults.Property(item, "probability"));
                        if (type == "score")
                        {
                            DecisionResults.Number(DecisionResults.Property(item, "value"));
                            if (DecisionResults.String(item, "label") is null) throw new JsonException("Missing score label.");
                        }
                        else if (!DecisionRequests.IsChoice(DecisionResults.Property(item, "value"))) throw new JsonException("Invalid choice value.");
                    }
                    break;
            }
        }
        return (body.Deserialize<OpenAIDecisionResponse>(DecisionRequests.Json) ?? throw new JsonException("Missing Decisions response."), body, upstreamHeaders);
    }

    internal static DecisionAnswer MapAnswer(DecisionQuestion question, OpenAIDecisionAnswer answer)
    {
        if (answer is OpenAIDecisionRefusalAnswer) return new DecisionRefusalAnswer();
        switch (question, answer)
        {
            case (DecisionBooleanQuestion { Criteria: null }, OpenAIDecisionPredicateAnswer predicate):
                return new DecisionBooleanAnswer { Probability = predicate.Probability };
            case (DecisionBooleanQuestion { Criteria: not null }, OpenAIDecisionChoiceAnswer boolean):
                return new DecisionBooleanAnswer { Probability = ChoiceDistribution(boolean, ["boolean:true", "boolean:false"])["boolean:true"] };
            case (DecisionChoiceQuestion q, OpenAIDecisionChoiceAnswer choice):
                var options = ChoiceDistribution(choice, q.Criteria.Keys.Select(x => "string:" + x));
                return new DecisionChoiceAnswer { Choice = choice.Choice.Value.GetString()!,
                    Probabilities = q.Criteria.Keys.ToDictionary(x => x, x => options["string:" + x], StringComparer.Ordinal) };
            case (DecisionScoreQuestion q, OpenAIDecisionScoreAnswer score):
                var indices = Enumerable.Range(0, q.Criteria.Count).ToDictionary(DecisionRequests.Index);
                var distribution = DecisionResults.Distribution(score.Probabilities.Select(x => (x.Label, x.Probability)), indices.Keys);
                return new DecisionScoreAnswer { Score = distribution.Sum(x => indices[x.Key] * x.Value), Probabilities = distribution };
            default: throw new JsonException("Decision answer does not match its question.");
        }
    }

    private static Dictionary<string, double> ChoiceDistribution(OpenAIDecisionChoiceAnswer answer, IEnumerable<string> expected)
    {
        var result = DecisionResults.Distribution(answer.Probabilities.Select(x => (DecisionRequests.ChoiceKey(x.Value), x.Probability)), expected);
        if (!result.ContainsKey(DecisionRequests.ChoiceKey(answer.Choice))) throw new JsonException("Unknown selected decision choice.");
        return result;
    }
}

internal static class DecisionResults
{
    internal static DecisionResponse Sdk(string provider, string model, JsonElement body, Dictionary<string, string?> headers,
        Dictionary<string, DecisionAnswer> answers, int input, int output, IEnumerable<object>? warnings = null)
        => new()
        {
            Answers = answers,
            Usage = new() { InputTokens = input, OutputTokens = output },
            Warnings = warnings ?? [],
            ProviderMetadata = new() { [provider] = body },
            Response = new()
            {
                Id = String(body, "id") ?? headers.GetValueOrDefault("x-request-id"),
                ModelId = model.StartsWith(provider + "/", StringComparison.OrdinalIgnoreCase) ? model : $"{provider}/{model}",
                Timestamp = DateTimeOffset.UtcNow,
                Body = body,
                Headers = headers
            }
        };

    internal static JsonElement Property(JsonElement root, string name) => root.ValueKind == JsonValueKind.Object
        && root.TryGetProperty(name, out var value) ? value : default;
    internal static string? String(JsonElement root, string name) => Property(root, name) is { ValueKind: JsonValueKind.String } value ? value.GetString() : null;
    internal static string RequiredModel(JsonElement body) => String(body, "model") is { } model && !string.IsNullOrWhiteSpace(model)
        ? model : throw new JsonException("A decision response requires a model.");
    internal static (int Input, int Output) Usage(JsonElement body)
    {
        var usage = Property(body, "usage");
        return (Integer(usage, "input_tokens"), Integer(usage, "output_tokens"));
    }
    internal static int Integer(JsonElement root, string name) => Property(root, name) is { ValueKind: JsonValueKind.Number } value
        && value.TryGetInt32(out var count) && count >= 0 ? count : throw new JsonException($"Invalid decision usage '{name}'.");
    internal static void ValidateDetails(JsonElement usage, string name, params string[] counters)
    {
        var details = Property(usage, name);
        if (details.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined) return;
        if (details.ValueKind != JsonValueKind.Object) throw new JsonException($"Invalid decision usage '{name}'.");
        foreach (var counter in counters)
            if (details.TryGetProperty(counter, out _)) Integer(details, counter);
    }
    internal static double Number(JsonElement value) => value.ValueKind == JsonValueKind.Number && value.TryGetDouble(out var number)
        && double.IsFinite(number) ? number : throw new JsonException("Expected a finite decision number.");
    internal static double Probability(JsonElement value) => CheckProbability(Number(value));
    private static double CheckProbability(double value) => double.IsFinite(value) && value is >= 0 and <= 1
        ? value : throw new JsonException("Decision probability must be in [0,1].");

    internal static Dictionary<string, double> Distribution(IEnumerable<(string Key, double Probability)> values, IEnumerable<string> keys)
    {
        var expected = keys.ToHashSet(StringComparer.Ordinal);
        var result = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var (key, probability) in values)
            if (!expected.Contains(key) || !result.TryAdd(key, CheckProbability(probability)))
                throw new JsonException("Unknown or duplicate decision probability key.");
        if (result.Count != expected.Count || Math.Abs(result.Values.Sum() - 1) > 1e-6)
            throw new JsonException("Incomplete or unnormalized decision probability distribution.");
        return result;
    }
}
