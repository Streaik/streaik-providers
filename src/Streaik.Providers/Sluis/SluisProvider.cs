using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;

namespace Streaik.Providers.Sluis;

public static class SluisRegistration
{
    public static IServiceCollection AddSluis(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, SluisProvider.Id, "https://api.sluis.ai/v1/");
        return services.AddProvider<SluisProvider>(SluisProvider.Id);
    }
}

internal sealed class SluisProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), IResponsesProvider, INativeChatOptionsProvider
{
    internal const string Id = "sluis";

    public async Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        return await NativeHttp.PostAsync<ResponseRequest, ResponseResult>(client, "responses", request, Auth(), headers, ResponseJson.Default, ct);
    }

    public async IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        await foreach (var part in NativeHttp.StreamAsync<ResponseRequest, ResponseStreamPart>(client, "responses", request, Auth(), headers,
            ResponseJson.Default, part => part.Type is "response.completed" or "response.incomplete" or "response.failed" or "error", ct))
            yield return part;
    }

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        (string Id, string Description)[] shortcuts =
        [
            ("auto", "Per-request routing. Sluis picks the best compliant target for the prompt."),
            ("code", "Coding and code review."), ("chat", "General conversation."), ("support", "Customer-support answers."),
            ("fast", "Low latency."), ("cheap", "Best value for bulk work."), ("agents", "Tool use and agentic workflows."),
            ("extract", "Structured extraction."), ("docs", "Document extraction and parsing."), ("longcontext", "Long-context analysis."),
            ("vision", "Image input."), ("translate", "Multilingual translation."), ("sovereign", "Open-weights only.")
        ];
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (id, description) in shortcuts)
        {
            var model = "sluis/" + id;
            seen.Add(model);
            yield return new() { Id = model, Name = id, Type = "language", OwnedBy = "Sluis", Description = description,
                Tags = id switch { "agents" => ["agent"], "translate" => ["translate"], _ => null } };
        }
        await foreach (var model in base.ListModelsAsync(ct))
            if (seen.Add(model.Id)) yield return model;
    }

    private Action<HttpRequestHeaders> Auth()
    {
        var key = ApiKey();
        return headers => headers.Authorization = new("Bearer", key);
    }
}
