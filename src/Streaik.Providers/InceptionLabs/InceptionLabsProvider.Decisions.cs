using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.Providers.Decisions;
using static Streaik.Providers.MediaHttp;
using Streaik.Vercel.Models;

namespace Streaik.Providers;

internal sealed partial class InceptionLabsProvider
{
    public async Task<DecisionResponse> DecideAsync(
        DecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        DecisionRequests.Validate(request, requiresInstructions: false);
        if (request.Model != "mercury-decide")
            throw new ProviderRequestException("Inception Labs Decisions supports model mercury-decide.");
        if (DecisionRequests.Options(request, Id).Any())
            throw new ProviderRequestException("Inception Labs Decisions does not support native request options.");

        var model = request.Model;
        var state = JsonNode.Parse(DecisionRequests.State(request).GetRawText());
        if (request.StateFormat == "sdk") state = MapInceptionDecisionMessages(state!);
        var questions = new JsonObject();
        foreach (var (id, question) in request.Questions)
        {
            var mapped = new JsonObject
            {
                ["type"] = question is DecisionBooleanQuestion ? "noul" : question.Type
            };
            if (question.Instructions is not null)
                mapped["instructions"] = JsonSerializer.SerializeToNode(question.Instructions);
            switch (question)
            {
                case DecisionBooleanQuestion boolean:
                    if (boolean.Criteria is not null)
                        mapped["criteria"] = JsonSerializer.SerializeToNode(boolean.Criteria);
                    break;
                case DecisionChoiceQuestion choice:
                    if (choice.Criteria.Count > 255)
                        throw new ProviderRequestException($"Inception choice question '{id}' supports 1 to 255 options.");
                    mapped["criteria"] = JsonSerializer.SerializeToNode(choice.Criteria);
                    break;
                case DecisionScoreQuestion score:
                    if (score.Criteria.Count > 10 || score.Criteria.Any(level => level is null))
                        throw new ProviderRequestException($"Inception score question '{id}' supports 1 to 10 nonnull levels.");
                    mapped["criteria"] = JsonSerializer.SerializeToNode(score.Criteria);
                    break;
                default:
                    throw new ProviderRequestException($"Inception Labs Decisions does not support question type '{question.Type}'.");
            }
            questions[id] = mapped;
        }

        var images = 0;
        ValidateInceptionDecisionMedia(state, true, ref images);
        foreach (var question in questions.Select(item => item.Value!.AsObject()))
        {
            ValidateInceptionDecisionMedia(question["instructions"], true, ref images);
            ValidateInceptionDecisionMedia(question["criteria"], false, ref images);
        }
        var payload = JsonSerializer.SerializeToElement(new JsonObject
        {
            ["model"] = model, ["state"] = state, ["questions"] = questions
        }, Json);
        if (System.Text.Encoding.UTF8.GetByteCount(payload.GetRawText()) > 20 * 1024 * 1024)
            throw new ProviderRequestException("Inception decision requests must not exceed 20 MiB.");
        using var client = CreateClient();
        var (body, upstream) = await NativeHttp.PostJsonAsync(client, "decisions", payload, ChatAuth(), headers, Json, ct);
        try
        {
            var result = MapInceptionDecisionResponse(body, request);
            var metadataHeaders = NativeHttp.SafeResponseMetadataHeaders(upstream);
            result.ProviderMetadata = NativeMediaOptions.Metadata(Id, body);
            result.Response = new()
            {
                ModelId = model, Timestamp = DateTime.UtcNow,
                Id = metadataHeaders.GetValueOrDefault("x-request-id"), Headers = metadataHeaders, Body = body
            };
            return result;
        }
        catch (Exception exception) when (exception is InvalidOperationException or KeyNotFoundException or FormatException or OverflowException)
        {
            throw new JsonException("Inception Labs returned an invalid decision response.", exception);
        }
    }

    private static JsonNode MapInceptionDecisionMessages(JsonNode state)
    {
        foreach (var message in state.AsArray().OfType<JsonObject>())
            if (message["content"] is JsonArray parts) MapInceptionDecisionParts(parts);
        return state;
    }

    private static void MapInceptionDecisionParts(JsonArray parts)
    {
        foreach (var part in parts.OfType<JsonObject>())
            switch (part["type"]?.GetValue<string>())
            {
                case "input_text":
                    part["type"] = "text";
                    break;
                case "input_image":
                    if (part["detail"] is not null && part["detail"]!.GetValue<string>() != "auto")
                        throw new ProviderRequestException("Inception Labs Decisions does not support image detail controls.");
                    if (part["image_url"] is not JsonValue value || !value.TryGetValue<string>(out var url))
                        throw new ProviderRequestException("Inception decision images require an inline image URL.");
                    part.Remove("detail");
                    part["type"] = "image_url";
                    part["image_url"] = new JsonObject { ["url"] = url };
                    break;
            }
    }

    private static void ValidateInceptionDecisionMedia(JsonNode? node, bool allowImages, ref int images)
    {
        if (node is JsonArray array)
        {
            foreach (var item in array)
                ValidateInceptionDecisionMedia(item, allowImages, ref images);
        }
        else if (node is JsonObject obj)
        {
            if (obj["type"] is JsonValue type && type.TryGetValue<string>(out var kind))
            {
                if (kind is "input_image" or "image" or "input_audio" or "audio" or "video" or "input_video" or "file" or "input_file")
                    throw new ProviderRequestException("Inception Labs Decisions accepts only chat-style image_url media parts; audio, video, and binary files are unsupported.");
                if (kind == "image_url")
                {
                    if (!allowImages)
                        throw new ProviderRequestException("Inception Labs decision images are supported in state and instructions, not criteria.");
                    if (obj["image_url"] is not JsonObject image
                        || image["url"] is not JsonValue urlValue || !urlValue.TryGetValue<string>(out var url))
                        throw new ProviderRequestException("Inception decision image_url parts require an object with a URL.");
                    if (obj.Count != 2 || image.Count != 1)
                        throw new ProviderRequestException("Inception decision images do not support additional fields or detail controls.");
                    var comma = url.IndexOf(',');
                    if (comma < 0 || !new[] { "data:image/png;base64", "data:image/jpeg;base64",
                        "data:image/webp;base64", "data:image/gif;base64" }.Contains(url[..comma], StringComparer.OrdinalIgnoreCase))
                        throw new ProviderRequestException("Inception decision images require inline base64 PNG, JPEG, WebP, or GIF data URLs; remote URLs are unsupported.");
                    byte[] data;
                    try { data = Convert.FromBase64String(url[(comma + 1)..]); }
                    catch (FormatException) { throw new ProviderRequestException("Inception decision image data must be base64."); }
                    if (data.Length is 0 or > 10 * 1024 * 1024)
                        throw new ProviderRequestException("Inception decision images must be nonempty and at most 10 MiB each.");
                    if (++images > 16)
                        throw new ProviderRequestException("Inception decision requests support at most 16 images across state and instructions.");
                    return;
                }
            }
            foreach (var property in obj)
                ValidateInceptionDecisionMedia(property.Value, allowImages, ref images);
        }
    }

    private static DecisionResponse MapInceptionDecisionResponse(JsonElement body, DecisionRequest request)
    {
        if (body.GetProperty("model").GetString() != "mercury-decide")
            throw new JsonException("Invalid decision model.");
        var native = body.GetProperty("answers");
        if (native.ValueKind != JsonValueKind.Object || native.EnumerateObject().Count() != request.Questions.Count
            || native.EnumerateObject().Select(item => item.Name).Distinct(StringComparer.Ordinal).Count() != request.Questions.Count)
            throw new JsonException("Answers must cover every question exactly once.");
        var answers = new Dictionary<string, DecisionAnswer>(StringComparer.Ordinal);
        foreach (var (id, question) in request.Questions)
        {
            var answer = native.GetProperty(id);
            if (answer.GetProperty("type").GetString() != (question is DecisionBooleanQuestion ? "noul" : question.Type))
                throw new JsonException("Decision answer type does not match question.");
            switch (question)
            {
                case DecisionBooleanQuestion:
                    answers[id] = new DecisionBooleanAnswer { Probability = ReadInceptionProbability(answer.GetProperty("noul")) };
                    break;
                case DecisionChoiceQuestion choice:
                    ReadInceptionProbability(answer.GetProperty("confidence"));
                    var distribution = ReadInceptionDistribution(answer.GetProperty("probabilities"), choice.Criteria.Keys);
                    var selected = answer.GetProperty("choice").GetString();
                    if (selected is null || !distribution.TryGetValue(selected, out var probability)
                        || probability + 1e-6 < distribution.Values.Max())
                        throw new JsonException("Selected choice must be a maximal option.");
                    answers[id] = new DecisionChoiceAnswer { Choice = selected, Probabilities = distribution };
                    break;
                case DecisionScoreQuestion score:
                    ReadInceptionProbability(answer.GetProperty("confidence"));
                    var keys = Enumerable.Range(0, score.Criteria.Count).Select(InceptionDecisionIndex).ToArray();
                    var probabilities = ReadInceptionDistribution(answer.GetProperty("probabilities"), keys);
                    var legend = answer.GetProperty("legend");
                    if (legend.ValueKind != JsonValueKind.Object || legend.EnumerateObject().Count() != keys.Length
                        || keys.Any(key => !legend.TryGetProperty(key, out var level)
                            || !JsonElement.DeepEquals(level, score.Criteria[int.Parse(key, CultureInfo.InvariantCulture)]!.Value)))
                        throw new JsonException("Score legend must match the requested structured levels.");
                    var value = answer.GetProperty("score").GetDouble();
                    var expected = probabilities.Sum(item => int.Parse(item.Key, CultureInfo.InvariantCulture) * item.Value);
                    if (!double.IsFinite(value) || value < 0 || value > keys.Length - 1 || Math.Abs(value - expected) > 1e-6)
                        throw new JsonException("Score must be the probability-weighted level position.");
                    answers[id] = new DecisionScoreAnswer { Score = value, Probabilities = probabilities };
                    break;
            }
        }
        var usage = body.GetProperty("usage");
        var input = usage.GetProperty("input_tokens").GetInt32();
        var output = usage.GetProperty("output_tokens").GetInt32();
        if (input < 0 || output < 0 || (long)input + output > int.MaxValue)
            throw new JsonException("Invalid decision usage.");
        return new() { Answers = answers, Usage = new() { InputTokens = input, OutputTokens = output } };
    }

    private static Dictionary<string, double> ReadInceptionDistribution(JsonElement value, IEnumerable<string> keys)
    {
        var expected = keys.ToHashSet(StringComparer.Ordinal);
        if (value.ValueKind != JsonValueKind.Object)
            throw new JsonException("Expected a probability map.");
        var result = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var property in value.EnumerateObject())
            if (!expected.Contains(property.Name) || !result.TryAdd(property.Name, ReadInceptionProbability(property.Value)))
                throw new JsonException("Unknown or duplicate probability key.");
        if (result.Count != expected.Count || Math.Abs(result.Values.Sum() - 1) > 1e-6)
            throw new JsonException("Incomplete or unnormalized probability distribution.");
        return result;
    }

    private static double ReadInceptionProbability(JsonElement value)
    {
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetDouble(out var number)
            || !double.IsFinite(number) || number is < 0 or > 1)
            throw new JsonException("Expected a finite probability in [0,1].");
        return number;
    }

    private static string InceptionDecisionIndex(int index) => index.ToString(CultureInfo.InvariantCulture);
}
