using System.IO.Compression;
using System.Runtime.CompilerServices;
using Streaik.OpenAI.Models.Decisions;
using Streaik.Vercel.Models;

namespace Streaik.Providers;

public sealed partial class EchoProvider : IDecisionsProvider, ISkillReadProvider, ISkillUploadProvider
{
    private const string DecisionModel = "echo-decision";
    private const string SkillId = "echo-fixture";
    private const string SkillName = "Echo fixture";
    private const string SkillDescription = "A static skill for testing the Echo provider.";
    private static readonly ProviderSkill Skill = new(SkillId, SkillName, SkillDescription, 0, "1", "1");

    public Task<OpenAIDecisionResponse> DecideAsync(OpenAIDecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        CheckModel(request.Model, DecisionModel, ct);
        return Task.FromResult(new OpenAIDecisionResponse { Model = "echo/" + request.Model, Answers = request.Questions.Select(question =>
        {
            OpenAIDecisionAnswer answer = question switch
            {
                OpenAIDecisionPredicateQuestion => new OpenAIDecisionPredicateAnswer { Probability = 0.5 },
                OpenAIDecisionChoiceQuestion { Choices.Count: > 0 } choice => new OpenAIDecisionChoiceAnswer
                    { Choice = choice.Choices[0].Value, Confidence = 1, Probabilities = choice.Choices.Select((x, index) =>
                        new OpenAIDecisionChoiceProbability { Value = x.Value, Probability = index == 0 ? 1 : 0 }).ToArray() },
                OpenAIDecisionScoreQuestion { Levels.Count: > 0 } score => new OpenAIDecisionScoreAnswer
                    { Score = 0, Confidence = 1, Probabilities = score.Levels.Select((level, index) =>
                        new OpenAIDecisionScoreProbability { Label = level.Label, Value = index, Probability = index == 0 ? 1 : 0 }).ToArray() },
                _ => new OpenAIDecisionRefusalAnswer()
            };
            answer.Name = question.Name;
            return answer;
        }).ToArray() });
    }

    public Task<DecisionResponse> DecideAsync(DecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        CheckModel(request.Model, DecisionModel, ct);
        return Task.FromResult(new DecisionResponse { Usage = new(), Response = new() { ModelId = "echo/" + request.Model },
            Answers = request.Questions.ToDictionary(pair => pair.Key, pair => (DecisionAnswer)(pair.Value switch
            {
                DecisionChoiceQuestion { Criteria.Count: > 0 } choice => new DecisionChoiceAnswer { Choice = choice.Criteria.Keys.First() },
                DecisionScoreQuestion { Criteria.Count: > 0 } => new DecisionScoreAnswer { Score = 0 },
                DecisionBooleanQuestion => new DecisionBooleanAnswer { Probability = 0.5 },
                DecisionBoxQuestion => new DecisionBoxAnswer { Box = new() { Top = 0, Left = 0, Bottom = 1, Right = 1 } },
                _ => new DecisionRefusalAnswer()
            })) });
    }

    public async IAsyncEnumerable<ProviderSkill> ListSkillsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        ct.ThrowIfCancellationRequested();
        yield return Skill;
    }

    public async IAsyncEnumerable<ProviderSkillVersion> ListSkillVersionsAsync(string skillId, [EnumeratorCancellation] CancellationToken ct)
    {
        await Task.CompletedTask;
        CheckSkill(skillId, null, ct);
        yield return new("echo-fixture-v1", SkillName, SkillDescription, 0, SkillId, "1");
    }

    public async Task<byte[]> DownloadSkillAsync(string skillId, string? version, CancellationToken ct)
    {
        CheckSkill(skillId, version, ct);
        using var output = new MemoryStream();
        using (var zip = new ZipArchive(output, ZipArchiveMode.Create, leaveOpen: true))
        {
            var entry = zip.CreateEntry("echo-fixture/SKILL.md");
            entry.LastWriteTime = new DateTimeOffset(1980, 1, 1, 0, 0, 0, TimeSpan.Zero);
            await using var file = entry.Open();
            await file.WriteAsync("---\nname: echo-fixture\ndescription: Static gateway contract fixture.\n---\nEcho does not execute or upload this skill.\n"u8.ToArray(), ct);
        }
        return output.ToArray();
    }

    public Task<ProviderSkill> UploadSkillAsync(byte[] archive, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        // The gateway has validated the archive. Acknowledge it without retaining any of its bytes.
        return Task.FromResult(Skill with { Description = "Fixture upload acknowledged; no content was stored." });
    }

    private static void CheckSkill(string skillId, string? version, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        if (skillId != SkillId || version is not (null or "1")) throw new ProviderRequestException("Unknown Echo skill or version.", 404);
    }
}
