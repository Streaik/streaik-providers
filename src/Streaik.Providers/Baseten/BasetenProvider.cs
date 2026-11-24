using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.OpenAI.Models.Decisions;
using Streaik.Providers.Decisions;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class BasetenProviderRegistration
{
    public static IServiceCollection AddBaseten(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, BasetenProvider.Id, "https://inference.baseten.co/v1/");
        return services.AddProvider<BasetenProvider>(BasetenProvider.Id);
    }
}

internal sealed class BasetenProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), INativeChatOptionsProvider, IDecisionsProvider
{
    internal const string Id = "baseten";
    private static readonly Dictionary<string, string> Decisions = new(StringComparer.Ordinal)
    {
        ["decision-models/baseten-elephant-9b-latest"] = "Elephant-1",
        ["decision-models/baseten-mammoth-27b-latest"] = "Mammoth-1"
    };

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(ConfiguredApiKey())) yield break;
        using var client = CreateClient();
        var body = await NativeHttp.GetAsync<JsonElement>(client, "models", Bearer(ApiKey()), ct);
        var listed = new HashSet<string>(StringComparer.Ordinal);
        foreach (var item in Array(Property(body, "data")))
        {
            if (String(item, "id") is not { Length: > 0 } id || !listed.Add(id)) continue;
            Decisions.TryGetValue(id, out var decision);
            var context = Property(item, "context_length");
            var pricing = Property(item, "pricing");
            yield return new Model
            {
                Id = id, Name = String(item, "name") ?? decision ?? id, OwnedBy = String(item, "owned_by") ?? Id,
                Description = String(item, "description"), Type = decision is null ? "language" : "decision",
                ContextWindow = context.ValueKind == JsonValueKind.Number && context.TryGetInt32(out var size) ? size : decision is null ? null : 16384,
                Pricing = Price(pricing, "input", "prompt") is { } input && Price(pricing, "output", "completion") is { } output
                    ? new() { Input = input, Output = output } : null
            };
        }
        foreach (var (id, name) in Decisions)
            if (listed.Add(id)) yield return new Model { Id = id, Name = name, OwnedBy = Id, Type = "decision", ContextWindow = 16384 };
    }

    private static decimal? Price(JsonElement pricing, string key, string alias)
    {
        var value = Property(pricing, key);
        if (value.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null) value = Property(pricing, alias);
        return value.ValueKind is JsonValueKind.Number or JsonValueKind.String
            && decimal.TryParse(value.ToString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var price) && price >= 0 ? price : null;
    }

    public async Task<DecisionResponse> DecideAsync(DecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        DecisionRequests.Validate(request);
        var input = DecisionRequests.State(request);
        var state = request.StateFormat == "sdk" ? State(input.Deserialize<OpenAIDecisionInput>(DecisionRequests.Json)!) : Copy(input);
        var model = ModelId(request.Model);
        var options = Extras(DecisionRequests.Options(request, Id).Select(p => KeyValuePair.Create(p.Name, p.Value)));
        var (body, upstream, answers) = await SendAsync(model, state, request.Questions, options, headers, ct);
        var (tokensIn, tokensOut) = DecisionResults.Usage(body);
        var result = DecisionResults.Sdk(Id, model, body, upstream, answers, tokensIn, tokensOut);
        if (String(body, "request_id") is { } id) result.Response!.Id = id;
        return result;
    }

    public async Task<OpenAIDecisionResponse> DecideAsync(OpenAIDecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        DecisionRequests.Validate(request);
        if (request.SafetyIdentifier is not null || request.PromptCacheKey is not null || request.PromptCacheOptions is not null)
            throw new ProviderRequestException("Baseten Decisions does not support safety identifiers or prompt caching settings.");
        var model = ModelId(request.Model);
        var questions = request.Questions.Select((q, i) => KeyValuePair.Create(QuestionId(i), Question(q))).ToDictionary();
        var (body, _, answers) = await SendAsync(model, State(request.Input), questions, Extras(request.AdditionalProperties ?? []), headers, ct);
        var native = Property(body, "answers");
        var (input, output) = DecisionResults.Usage(body);
        if ((long)input + output > int.MaxValue) throw new JsonException("Invalid Baseten token total.");
        return new OpenAIDecisionResponse
        {
            Model = model, Usage = new() { InputTokens = input, OutputTokens = output, TotalTokens = input + output },
            Answers = request.Questions.Select((q, i) => OpenAIAnswer(q, answers[QuestionId(i)], Property(native, QuestionId(i)))).ToArray(),
            AdditionalProperties = new()
            {
                ["providerMetadata"] = JsonSerializer.SerializeToElement(new Dictionary<string, JsonElement> { [Id] = body }, Json),
                ["warnings"] = JsonSerializer.SerializeToElement(new[]
                {
                    new { type = "other", message = "Baseten System One reports one output token per question." }
                }, Json)
            }
        };
    }

    private static string ModelId(string id) => Decisions.ContainsKey(id) ? id : throw new ProviderRequestException("Unknown Baseten Decisions model.");
    private static string QuestionId(int i) => "question-" + DecisionRequests.Index(i);
    private static string OptionId(int i) => "option-" + DecisionRequests.Index(i);
    private static string Type(DecisionQuestion question) => question is DecisionBooleanQuestion ? "noul" : question.Type;
    private static string Text(DecisionInput value) => value.Value.ValueKind == JsonValueKind.String ? value.Value.GetString()! : value.Value.GetRawText();
    private static string ChoiceLabel(OpenAIDecisionChoiceValue value) => value.Value.ValueKind == JsonValueKind.String ? value.Value.GetString()! : value.Value.GetRawText();
    private static string Describe(string label, string? description) => string.IsNullOrEmpty(description) ? label : $"{label}: {description}";

    private static JsonObject Extras(IEnumerable<KeyValuePair<string, JsonElement>> extras)
    {
        var options = new JsonObject();
        foreach (var (key, value) in extras)
        {
            if (key.ToLowerInvariant() is "model" or "state" or "questions" or "input" or "stream")
                throw new ProviderRequestException("Baseten options cannot override the decision request or enable streaming.");
            if (key == "service_tier" && (value.ValueKind != JsonValueKind.String || value.GetString() is not ("default" or "priority")))
                throw new ProviderRequestException("Baseten service_tier must be 'default' or 'priority'.");
            options[key] = Copy(value);
        }
        return options;
    }

    private async Task<(JsonElement Body, Dictionary<string, string?> Headers, Dictionary<string, DecisionAnswer> Answers)> SendAsync(
        string model, JsonNode? state, Dictionary<string, DecisionQuestion> questions, JsonObject payload, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var native = new JsonObject();
        foreach (var (id, question) in questions)
        {
            var mapped = new JsonObject { ["type"] = Type(question), ["instructions"] = Text(question.Instructions!) };
            switch (question)
            {
                case DecisionBooleanQuestion { Criteria: { } criteria }:
                    var boolean = new JsonObject();
                    if (criteria.True is { } yes) boolean["true"] = Text(yes);
                    if (criteria.False is { } no) boolean["false"] = Text(no);
                    if (boolean.Count > 0) mapped["criteria"] = boolean;
                    break;
                case DecisionChoiceQuestion choice:
                    var options = new JsonObject();
                    foreach (var (key, value) in choice.Criteria) options[key] = value is null ? key : Text(value);
                    mapped["criteria"] = options;
                    break;
                case DecisionScoreQuestion score:
                    mapped["criteria"] = new JsonArray(score.Criteria.Select(level => (JsonNode?)JsonValue.Create(level is null
                        ? throw new ProviderRequestException("Baseten requires text for every score level.") : Text(level))).ToArray());
                    break;
            }
            native[id] = mapped;
        }
        payload["model"] = model;
        payload["state"] = state ?? throw new ProviderRequestException("Baseten decision state cannot be null.");
        payload["questions"] = native;
        using var client = CreateClient();
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, "systemone", payload, Bearer(ApiKey()), headers, Json, ct);
        var answers = Property(body, "answers");
        if (DecisionResults.RequiredModel(body) != model || answers.ValueKind != JsonValueKind.Object
            || answers.EnumerateObject().Count() != questions.Count) throw new JsonException("Invalid Baseten decision response.");
        return (body, upstream, questions.ToDictionary(q => q.Key, q => Answer(q.Value, Property(answers, q.Key)), StringComparer.Ordinal));
    }

    private static DecisionAnswer Answer(DecisionQuestion question, JsonElement native)
    {
        if (String(native, "type") != Type(question)) throw new JsonException("Baseten answer type does not match the question.");
        if (question is DecisionBooleanQuestion) return new DecisionBooleanAnswer { Probability = Probability(native, "noul") };
        Probability(native, "confidence");
        var raw = Property(native, "probabilities");
        if (raw.ValueKind != JsonValueKind.Object) throw new JsonException("Missing Baseten probabilities.");
        var keys = question is DecisionChoiceQuestion choice ? choice.Criteria.Keys.ToArray()
            : Enumerable.Range(0, ((DecisionScoreQuestion)question).Criteria.Count).Select(DecisionRequests.Index).ToArray();
        var distribution = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var p in raw.EnumerateObject())
            if (!keys.Contains(p.Name) || !distribution.TryAdd(p.Name, DecisionResults.Probability(p.Value)))
                throw new JsonException("Invalid Baseten probability key.");
        if (distribution.Count != keys.Length || Math.Abs(distribution.Values.Sum() - 1) > Math.Max(.001, keys.Length * .00005))
            throw new JsonException("Incomplete or unnormalized Baseten probability distribution.");
        if (question is DecisionChoiceQuestion)
        {
            var selected = String(native, "choice");
            if (selected is null || !distribution.TryGetValue(selected, out var p) || p + 1e-6 < distribution.Values.Max())
                throw new JsonException("Invalid Baseten choice.");
            return new DecisionChoiceAnswer { Choice = selected, Probabilities = distribution };
        }
        var score = DecisionResults.Number(Property(native, "score"));
        var legend = Property(native, "legend");
        if (legend.ValueKind != JsonValueKind.Object || legend.EnumerateObject().Count() != keys.Length
            || keys.Where((key, i) => String(legend, key) != Text(((DecisionScoreQuestion)question).Criteria[i]!)).Any()
            || score < 0 || score > keys.Length - 1
            || Math.Abs(score - keys.Select((key, i) => i * distribution[key]).Sum()) > Math.Max(.0001, (keys.Length * (keys.Length - 1) / 2d + 1) * .00005))
            throw new JsonException("Invalid Baseten score or legend.");
        return new DecisionScoreAnswer { Score = score, Probabilities = distribution };
    }

    private static double Probability(JsonElement body, string key) => DecisionResults.Probability(Property(body, key));
    private static bool BooleanChoice(OpenAIDecisionChoiceQuestion q) => q.Choices.Count == 2
        && q.Choices.Select(x => x.Value.Value.ValueKind).ToHashSet().SetEquals([JsonValueKind.True, JsonValueKind.False]);

    private static DecisionQuestion Question(OpenAIDecisionQuestion q)
    {
        DecisionInput? Criterion(OpenAIDecisionChoiceQuestion c, bool value)
            => c.Choices.Single(option => option.Value.Value.GetBoolean() == value).Description is { } text ? new(text) : null;
        return q switch
        {
            OpenAIDecisionPredicateQuestion => new DecisionBooleanQuestion { Instructions = q.Instructions },
            OpenAIDecisionChoiceQuestion c when BooleanChoice(c) => new DecisionBooleanQuestion { Instructions = q.Instructions,
                Criteria = new() { True = Criterion(c, true), False = Criterion(c, false) } },
            OpenAIDecisionChoiceQuestion c => new DecisionChoiceQuestion { Instructions = q.Instructions,
                Criteria = c.Choices.Select((o, i) => KeyValuePair.Create(OptionId(i), (DecisionInput?)new DecisionInput(Describe(ChoiceLabel(o.Value), o.Description)))).ToDictionary() },
            OpenAIDecisionScoreQuestion s => new DecisionScoreQuestion { Instructions = q.Instructions,
                Criteria = s.Levels.Select(l => (DecisionInput?)new DecisionInput(Describe(l.Label, l.Description))).ToArray() },
            _ => throw new ProviderRequestException("Unsupported Baseten decision question.")
        };
    }

    private static OpenAIDecisionAnswer OpenAIAnswer(OpenAIDecisionQuestion question, DecisionAnswer answer, JsonElement native)
    {
        OpenAIDecisionAnswer result = (question, answer) switch
        {
            (OpenAIDecisionPredicateQuestion, DecisionBooleanAnswer b) => new OpenAIDecisionPredicateAnswer { Probability = b.Probability },
            (OpenAIDecisionChoiceQuestion c, DecisionBooleanAnswer b) => new OpenAIDecisionChoiceAnswer
            {
                Choice = c.Choices.Single(o => o.Value.Value.GetBoolean() == (b.Probability >= .5)).Value,
                Confidence = Math.Max(b.Probability, 1 - b.Probability),
                Probabilities = c.Choices.Select(o => new OpenAIDecisionChoiceProbability { Value = o.Value, Probability = o.Value.Value.GetBoolean() ? b.Probability : 1 - b.Probability }).ToArray()
            },
            (OpenAIDecisionChoiceQuestion c, DecisionChoiceAnswer a) => new OpenAIDecisionChoiceAnswer
            {
                Choice = c.Choices.Where((_, i) => OptionId(i) == a.Choice).Single().Value, Confidence = Probability(native, "confidence"),
                Probabilities = c.Choices.Select((o, i) => new OpenAIDecisionChoiceProbability { Value = o.Value, Probability = a.Probabilities![OptionId(i)] }).ToArray()
            },
            (OpenAIDecisionScoreQuestion s, DecisionScoreAnswer a) => new OpenAIDecisionScoreAnswer
            {
                Score = a.Score, Confidence = Probability(native, "confidence"),
                Probabilities = s.Levels.Select((l, i) => new OpenAIDecisionScoreProbability { Label = l.Label, Value = i, Probability = a.Probabilities![DecisionRequests.Index(i)] }).ToArray()
            },
            _ => throw new JsonException("Invalid Baseten answer type.")
        };
        result.Name = question.Name;
        return result;
    }

    private static JsonNode State(OpenAIDecisionInput input)
    {
        if (input.Text is { } text) return JsonValue.Create(text)!;
        var messages = new JsonArray();
        foreach (var message in input.Messages ?? [])
            messages.Add(new JsonObject { ["role"] = "user", ["content"] = message.Content.Text is { } value ? JsonValue.Create(value)
                : new JsonArray((message.Content.Parts ?? []).Select(p => (JsonNode)(p switch
                {
                    OpenAIDecisionInputText t => new JsonObject { ["type"] = "text", ["text"] = t.Text },
                    OpenAIDecisionInputImage { Detail: null } i => new JsonObject { ["type"] = "image_url", ["image_url"] = new JsonObject { ["url"] = i.ImageUrl } },
                    _ => throw new ProviderRequestException("Unsupported Baseten Decisions input part or image setting.")
                })).ToArray()) });
        return new JsonObject { ["messages"] = messages };
    }
}
