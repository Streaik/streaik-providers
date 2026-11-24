using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.OpenAI.Models;
using Streaik.OpenAI.Models.Decisions;
using Streaik.Providers.Decisions;
using Streaik.Responses;
using Streaik.Responses.Streaming;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class CelerisProviderRegistration
{
    public static IServiceCollection AddCeleris(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, CelerisProvider.Id, "https://inference.celeris.ai/");
        return services.AddProvider<CelerisProvider>(CelerisProvider.Id);
    }
}

internal sealed class CelerisProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), INativeChatOptionsProvider, IResponsesProvider, IDecisionsProvider
{
    internal const string Id = "celeris";
    private const string DecisionModel = "celeris-1-decision";
    private const string DecisionAlias = "celeris-1.1-decision";

    private static string ModelSegment(string? model)
    {
        if (string.IsNullOrEmpty(model) || model is "." or ".." || !model.All(ch => char.IsAsciiLetterOrDigit(ch) || ch is '-' or '_' or '.'))
            throw new ProviderRequestException("Invalid Celeris model ID.");
        return model == DecisionAlias ? DecisionModel : model;
    }

    public override async Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        return await NativeHttp.PostAsync<ChatCompletionOptions, ChatCompletion>(client, $"{ModelSegment(request.Model)}/v1/chat/completions",
            request, Bearer(ApiKey()), headers, Json, ct);
    }

    public override async IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        await foreach (var part in NativeHttp.StreamAsync<ChatCompletionOptions, ChatCompletionUpdate>(client,
            $"{ModelSegment(request.Model)}/v1/chat/completions", request, Bearer(ApiKey()), headers, Json, _ => false, ct)) yield return part;
    }

    public async Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        return await NativeHttp.PostAsync<ResponseRequest, ResponseResult>(client, $"{ModelSegment(request.Model)}/v1/responses",
            request, Bearer(ApiKey()), headers, ResponseJson.Default, ct);
    }

    public async IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        await foreach (var part in NativeHttp.StreamAsync<ResponseRequest, ResponseStreamPart>(client,
            $"{ModelSegment(request.Model)}/v1/responses", request, Bearer(ApiKey()), headers, ResponseJson.Default,
            part => part.Type is "response.completed" or "response.incomplete" or "response.failed" or "error", ct)) yield return part;
    }

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (ConfiguredApiKey() is not { } key || string.IsNullOrWhiteSpace(key)) yield break;
        using var client = CreateClient();
        var models = new Dictionary<string, Model>(StringComparer.Ordinal);
        foreach (var segment in new[] { "celeris-1", DecisionModel })
        {
            var catalog = await NativeHttp.GetAsync<JsonElement>(client, $"{segment}/v1/models", Bearer(key), ct);
            if (Property(catalog, "data") is not { ValueKind: JsonValueKind.Array } data)
                throw new JsonException("Celeris model catalog must contain a data array.");
            foreach (var item in data.EnumerateArray())
            {
                var model = item.Deserialize<Model>(Json) ?? throw new JsonException("Invalid Celeris model.");
                if (string.IsNullOrWhiteSpace(model.Id)) throw new JsonException("Celeris model IDs cannot be empty.");
                if (model.Id == DecisionModel) model.Id = DecisionAlias;
                if (string.IsNullOrWhiteSpace(model.Name)) model.Name = model.Id;
                if (string.IsNullOrWhiteSpace(model.OwnedBy)) model.OwnedBy = Id;
                model.Type = segment == DecisionModel ? "decision" : model.Type ?? "language";
                models[model.Id] = model;
            }
        }
        models.TryAdd("celeris-1", new() { Id = "celeris-1", Name = "Celeris 1", OwnedBy = Id, Type = "language" });
        models.TryAdd(DecisionAlias, new() { Id = DecisionAlias, Name = "Celeris 1.1 Decision", OwnedBy = Id, Type = "decision" });
        foreach (var model in models.Values) yield return model;
    }

    public async Task<OpenAIDecisionResponse> DecideAsync(OpenAIDecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var native = ModelSegment(request.Model);
        var upstream = new OpenAIDecisionRequest { Model = native, Input = request.Input, Questions = request.Questions,
            SafetyIdentifier = request.SafetyIdentifier, AdditionalProperties = request.AdditionalProperties };
        using var client = CreateClient();
        var result = await OpenAICompatibleDecisions.DecideAsync(client, $"{native}/v1/decisions", ApiKey(), upstream, headers, ct);
        result.Model = request.Model;
        return result;
    }

    public async Task<DecisionResponse> DecideAsync(DecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        DecisionRequests.Validate(request);
        var native = ModelSegment(request.Model);
        if (DecisionRequests.Options(request, Id).Any()) throw new ProviderRequestException("Celeris System One does not support provider options.");
        var questions = new JsonObject();
        foreach (var (id, question) in request.Questions)
        {
            if (question.Instructions?.Value is not { ValueKind: JsonValueKind.String } instructions)
                throw new ProviderRequestException("Celeris System One requires text instructions.");
            var mapped = new JsonObject { ["type"] = Kind(question), ["instructions"] = instructions.GetString() };
            switch (question)
            {
                case DecisionBooleanQuestion { Criteria: null }: break;
                case DecisionChoiceQuestion choice when choice.Criteria.Count >= 2: mapped["criteria"] = JsonSerializer.SerializeToNode(choice.Criteria, Json); break;
                case DecisionScoreQuestion score when score.Criteria.Count >= 2: mapped["criteria"] = JsonSerializer.SerializeToNode(score.Criteria, Json); break;
                default: throw new ProviderRequestException("Celeris requires choice/score criteria with at least two entries and no boolean criteria.");
            }
            questions[id] = mapped;
        }
        var payload = new JsonObject { ["model"] = native, ["state"] = Copy(DecisionRequests.State(request)), ["questions"] = questions };
        using var client = CreateClient();
        var (body, responseHeaders) = await NativeHttp.PostJsonAsync(client, $"{native}/v1/systemone", payload, Bearer(ApiKey()), headers, Json, ct);
        if (DecisionResults.RequiredModel(body) != native) throw new JsonException("Celeris System One model mismatch.");
        var raw = Property(body, "answers");
        if (raw.ValueKind != JsonValueKind.Object || raw.EnumerateObject().Select(x => x.Name).Distinct(StringComparer.Ordinal).Count() != request.Questions.Count
            || raw.EnumerateObject().Count() != request.Questions.Count) throw new JsonException("Expected one answer per Celeris question.");
        var answers = request.Questions.ToDictionary(entry => entry.Key, entry => Answer(entry.Value, Property(raw, entry.Key)), StringComparer.Ordinal);
        var (input, output) = DecisionResults.Usage(body);
        return DecisionResults.Sdk(Id, request.Model, body, responseHeaders, answers, input, output);
    }

    private static string Kind(DecisionQuestion question) => question is DecisionBooleanQuestion ? "noul" : question.Type;

    private static DecisionAnswer Answer(DecisionQuestion question, JsonElement answer)
    {
        if (String(answer, "type") != Kind(question)) throw new JsonException("Celeris answer type does not match the question.");
        switch (question)
        {
            case DecisionBooleanQuestion: return new DecisionBooleanAnswer { Probability = DecisionResults.Probability(Property(answer, "noul")) };
            case DecisionChoiceQuestion choice:
                var selected = String(answer, "choice");
                if (selected is null || !choice.Criteria.ContainsKey(selected)) throw new JsonException("Unknown Celeris choice.");
                var options = Distribution(answer, choice.Criteria.Keys);
                if (options is not null && options[selected] + 1e-6 < options.Values.Max()) throw new JsonException("Celeris choice was not the most probable.");
                return new DecisionChoiceAnswer { Choice = selected, Probabilities = options };
            case DecisionScoreQuestion score:
                var value = DecisionResults.Number(Property(answer, "score"));
                if (value < 0 || value > score.Criteria.Count - 1) throw new JsonException("Invalid Celeris score.");
                var indices = Enumerable.Range(0, score.Criteria.Count).ToDictionary(DecisionRequests.Index);
                var probabilities = Distribution(answer, indices.Keys);
                if (probabilities is not null && Math.Abs(value - probabilities.Sum(x => indices[x.Key] * x.Value)) > 1e-6)
                    throw new JsonException("Celeris score does not match its distribution.");
                return new DecisionScoreAnswer { Score = value, Probabilities = probabilities };
            default: throw new JsonException("Unsupported Celeris question type.");
        }
    }

    private static Dictionary<string, double>? Distribution(JsonElement answer, IEnumerable<string> keys)
    {
        if (!answer.TryGetProperty("probabilities", out var raw)) return null;
        if (raw.ValueKind != JsonValueKind.Object) throw new JsonException("Expected a Celeris probability map.");
        return DecisionResults.Distribution(raw.EnumerateObject().Select(x => (x.Name, DecisionResults.Probability(x.Value))), keys);
    }
}
