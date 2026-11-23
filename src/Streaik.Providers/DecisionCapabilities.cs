using Streaik.OpenAI.Models.Decisions;
using Streaik.Vercel.Models;

namespace Streaik.Providers;

public interface IDecisionsProvider
{
    Task<OpenAIDecisionResponse> DecideAsync(OpenAIDecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
    Task<DecisionResponse> DecideAsync(DecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct);
}
