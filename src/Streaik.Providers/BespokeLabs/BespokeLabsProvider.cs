using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.OpenAI.Models.Decisions;
using Streaik.Providers.Decisions;
using Streaik.Vercel.Models;
using static Streaik.Providers.Decisions.DecisionResults;

namespace Streaik.Providers;

public static class BespokeLabsProviderRegistration
{
    public static IServiceCollection AddBespokeLabs(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, BespokeLabsProvider.Id, "https://api.bespokelabs.ai/");
        return services.AddProvider<BespokeLabsProvider>(BespokeLabsProvider.Id);
    }
}

internal sealed class BespokeLabsProvider(IHttpClientFactory clients, IProviderCredentials credentials) : IModelListingProvider, IDecisionsProvider
{
    internal const string Id = "bespokelabs";
    private static readonly JsonSerializerOptions Json = DecisionRequests.Json;
    private const int MaxCriteria = 255;
    private const int MaxQuestions = 64;
    private const int MaxRequestBytes = 1024 * 1024;

    public IAsyncEnumerable<Model> ListModelsAsync(CancellationToken ct) => StaticModelListing.ListAsync(Id, credentials.GetApiKey(Id), ct);

    public async Task<OpenAIDecisionResponse> DecideAsync(OpenAIDecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        DecisionRequests.Validate(request);
        if (request.SafetyIdentifier is not null || request.PromptCacheKey is not null || request.PromptCacheOptions is not null
            || request.AdditionalProperties?.Count > 0)
            throw new ProviderRequestException("BespokeLabs does not support additional decision or prompt cache options.");
        if (request.Input.Items?.Any(item => item is not OpenAIDecisionInputMessage message || Unsupported(message.Content.Parts)) == true)
            throw new ProviderRequestException("BespokeLabs Decisions supports text and JSON state, not image or tool input.");

        var questions = request.Questions.Select((question, i) => new Question($"question-{DecisionRequests.Index(i)}", question)).ToArray();
        var result = await SendAsync(request.Model, JsonSerializer.SerializeToElement(request.Input, Json), questions, headers, ct);
        return new() { Model = request.Model, Answers = result.Answers,
            Usage = new() { InputTokens = result.Input, OutputTokens = result.Output, TotalTokens = result.Input + result.Output },
            AdditionalProperties = new() { ["providerMetadata"] = JsonSerializer.SerializeToElement(new Dictionary<string, JsonElement> { [Id] = result.Body }, Json) } };
    }

    public async Task<DecisionResponse> DecideAsync(DecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        DecisionRequests.Validate(request, requiresInstructions: false);
        if (DecisionRequests.Options(request, Id).Any()) throw new ProviderRequestException("BespokeLabs does not support additional decision options.");
        var questions = request.Questions.Select(pair => new Question(pair.Key, DecisionRequests.MapQuestion(pair.Key, pair.Value),
            pair.Value is DecisionBooleanQuestion { Criteria: not null })).ToArray();
        var result = await SendAsync(request.Model, DecisionRequests.State(request, allowImages: false), questions, headers, ct);
        var answers = result.Answers.ToDictionary(answer => answer.Name!, answer => OpenAICompatibleDecisions.MapAnswer(request.Questions[answer.Name!], answer),
            StringComparer.Ordinal);
        return Sdk(Id, request.Model, result.Body, result.Headers, answers, result.Input, result.Output);
    }

    private async Task<NativeResult> SendAsync(string model, JsonElement state, Question[] questions,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (model is not ("nimble-latest" or "nimble-v4-max" or "nimble-v4" or "nimble-v3"))
            throw new ProviderRequestException("Unsupported BespokeLabs decision model.");
        if (state.ValueKind is not (JsonValueKind.String or JsonValueKind.Object or JsonValueKind.Array))
            throw new ProviderRequestException("BespokeLabs decision state must be text, an object, or an array.");
        if (questions.Length is < 1 or > MaxQuestions) throw new ProviderRequestException("BespokeLabs requires 1–64 questions.");

        var nativeQuestions = new JsonObject();
        foreach (var question in questions) nativeQuestions.Add(question.Id, BuildQuestion(question));
        var payload = new JsonObject { ["model"] = model, ["state"] = MediaHttp.Copy(state), ["questions"] = nativeQuestions };
        if (Encoding.UTF8.GetByteCount(payload.ToJsonString(Json)) > MaxRequestBytes)
            throw new ProviderRequestException("BespokeLabs decision request exceeds 1 MiB.");

        using var client = clients.CreateClient(Id);
        var key = MediaHttp.RequiredKey(credentials, Id);
        var (body, upstream) = await SendWithFallbackAsync();
        if (RequiredModel(body) != model) throw new JsonException("BespokeLabs returned a different decision model.");
        var rawAnswers = Property(body, "answers");
        if (rawAnswers.ValueKind != JsonValueKind.Object || rawAnswers.EnumerateObject().Count() != questions.Length)
            throw new JsonException("BespokeLabs must return one answer per question.");
        var answers = questions.Select(question => ReadAnswer(Property(rawAnswers, question.Id), question)).ToArray();
        var (input, output) = Usage(body);
        if (input > int.MaxValue - output) throw new JsonException("Invalid BespokeLabs decision token total.");
        return new(body, NativeHttp.SafeResponseMetadataHeaders(upstream), answers, input, output);

        async Task<(JsonElement Body, Dictionary<string, string?> Headers)> SendWithFallbackAsync()
        {
            try { return await SendToAsync("v1/systemone"); }
            catch (HttpRequestException error) when (error.StatusCode == HttpStatusCode.NotFound)
            { return await SendToAsync("v1/nimble/systemone"); }
        }

        Task<(JsonElement Body, Dictionary<string, string?> Headers)> SendToAsync(string endpoint)
            => NativeHttp.PostJsonAsync(client, endpoint, payload, h => h.Authorization = new("Bearer", key), headers, Json, ct);
    }

    private static JsonObject BuildQuestion(Question question)
    {
        var mapped = new JsonObject { ["type"] = question.NativeType };
        if (question.Contract.InstructionContent is { } instructions)
        {
            if (Unsupported(instructions.Parts)) throw new ProviderRequestException("BespokeLabs decision instructions do not support images or prompt caching.");
            mapped["instructions"] = instructions.Text ?? JsonSerializer.Serialize(instructions, Json);
        }
        switch (question.Contract)
        {
            case OpenAIDecisionPredicateQuestion: break;
            case OpenAIDecisionChoiceQuestion choice when question.BooleanCriteria:
                var boolean = new JsonObject();
                foreach (var option in choice.Choices)
                    if (option.Description is not null) boolean[option.Value.Value.GetRawText()] = option.Description;
                if (boolean.Count > 0) mapped["criteria"] = boolean;
                break;
            case OpenAIDecisionChoiceQuestion choice when choice.Choices.Count is >= 2 and <= MaxCriteria:
                var criteria = new JsonObject();
                for (var i = 0; i < choice.Choices.Count; i++) criteria[OptionId(i)] = choice.Choices[i].Description;
                mapped["criteria"] = criteria;
                break;
            case OpenAIDecisionScoreQuestion score when score.Levels.Count is >= 2 and <= MaxCriteria:
                mapped["criteria"] = new JsonArray(score.Levels.Select(level => (JsonNode?)JsonValue.Create(level.Description ?? level.Label)).ToArray());
                break;
            default: throw new ProviderRequestException($"BespokeLabs question '{question.Id}' requires 2–255 criteria.");
        }
        return mapped;
    }

    private static OpenAIDecisionAnswer ReadAnswer(JsonElement answer, Question question)
    {
        if (String(answer, "type") != question.NativeType) throw new JsonException("BespokeLabs answer type does not match its question.");
        switch (question.Contract)
        {
            case OpenAIDecisionPredicateQuestion:
                return new OpenAIDecisionPredicateAnswer { Name = question.Contract.Name, Probability = Probability(Property(answer, "noul")) };
            case OpenAIDecisionChoiceQuestion choice when question.BooleanCriteria:
                var p = Probability(Property(answer, "noul"));
                return new OpenAIDecisionChoiceAnswer { Name = choice.Name, Choice = p >= .5, Confidence = Math.Max(p, 1 - p),
                    Probabilities = [new() { Value = true, Probability = p }, new() { Value = false, Probability = 1 - p }] };
            case OpenAIDecisionChoiceQuestion choice:
                var keys = Enumerable.Range(0, choice.Choices.Count).Select(OptionId).ToArray();
                var choices = ReadDistribution(answer, keys);
                var selected = System.Array.IndexOf(keys, String(answer, "choice"));
                if (selected < 0 || choices[keys[selected]] + 1e-6 < choices.Values.Max())
                    throw new JsonException("BespokeLabs selected a choice that was not a maximal option.");
                return new OpenAIDecisionChoiceAnswer { Name = choice.Name, Choice = choice.Choices[selected].Value,
                    Confidence = Probability(Property(answer, "confidence")),
                    Probabilities = choice.Choices.Select((option, i) => new OpenAIDecisionChoiceProbability
                        { Value = option.Value, Probability = choices[keys[i]] }).ToArray() };
            case OpenAIDecisionScoreQuestion score:
                var levels = Enumerable.Range(0, score.Levels.Count).Select(DecisionRequests.Index).ToArray();
                var probabilities = ReadDistribution(answer, levels);
                var weighted = Number(Property(answer, "score"));
                var legend = Property(answer, "legend");
                if (weighted < 0 || weighted > levels.Length - 1 || Math.Abs(weighted - levels.Select((key, i) => i * probabilities[key]).Sum()) > 1e-5
                    || legend.ValueKind != JsonValueKind.Object || legend.EnumerateObject().Count() != levels.Length
                    || levels.Any(key => String(legend, key) is null)) throw new JsonException("Invalid BespokeLabs score or legend.");
                return new OpenAIDecisionScoreAnswer { Name = score.Name, Score = weighted, Confidence = Probability(Property(answer, "confidence")),
                    Probabilities = score.Levels.Select((level, i) => new OpenAIDecisionScoreProbability
                        { Label = level.Label, Value = i, Probability = probabilities[levels[i]] }).ToArray() };
            default: throw new JsonException("Unexpected BespokeLabs decision question.");
        }
    }

    private static Dictionary<string, double> ReadDistribution(JsonElement answer, string[] keys)
        => Property(answer, "probabilities") is { ValueKind: JsonValueKind.Object } values
            ? Distribution(values.EnumerateObject().Select(p => (p.Name, Probability(p.Value))), keys)
            : throw new JsonException("Missing BespokeLabs decision probabilities.");

    private static bool Unsupported(IReadOnlyList<OpenAIDecisionInputPart>? parts)
        => parts?.Any(part => part is OpenAIDecisionInputImage or OpenAIDecisionInputText { PromptCacheBreakpoint: not null }) == true;
    private static string OptionId(int index) => "option-" + DecisionRequests.Index(index);

    private sealed record Question(string Id, OpenAIDecisionQuestion Contract, bool BooleanCriteria = false)
    {
        public string NativeType => BooleanCriteria || Contract is OpenAIDecisionPredicateQuestion ? "noul" : Contract.Type;
    }

    private sealed record NativeResult(JsonElement Body, Dictionary<string, string?> Headers, OpenAIDecisionAnswer[] Answers, int Input, int Output);
}
