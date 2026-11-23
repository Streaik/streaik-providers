using Streaik.OpenAI.Models.Decisions;
using Streaik.Vercel.Models;
using Streaik.Providers.Decisions;

namespace Streaik.Providers.OpenAI;

public sealed partial class OpenAIProvider
{
    public async Task<OpenAIDecisionResponse> DecideAsync(OpenAIDecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        return await OpenAICompatibleDecisions.DecideAsync(client, "v1/decisions", RequireKey(), request, headers, ct);
    }

    public async Task<DecisionResponse> DecideAsync(DecisionRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        return await OpenAICompatibleDecisions.DecideAsync(client, "v1/decisions", RequireKey(), Id, request, headers, ct);
    }
}
