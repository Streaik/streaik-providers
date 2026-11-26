using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class InceptionLabsProviderRegistration
{
    public static IServiceCollection AddInceptionLabs(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, InceptionLabsProvider.Id, "https://api.inceptionlabs.ai/v1/");
        return services.AddProvider<InceptionLabsProvider>(InceptionLabsProvider.Id);
    }
}

internal sealed partial class InceptionLabsProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), INativeChatOptionsProvider, IDecisionsProvider
{
    internal const string Id = "inceptionlabs";

    private static string ChatPath(string? model) => model switch
    {
        string m when m.Equals("mercury-edit/apply", StringComparison.OrdinalIgnoreCase) => "apply/completions",
        string m when m.StartsWith("mercury-edit", StringComparison.OrdinalIgnoreCase) => "edit/completions",
        _ => "chat/completions"
    };

    public override async Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        PrepareCompatibleRequest(request);
        return await NativeHttp.PostAsync<Dictionary<string, JsonElement>, ChatCompletion>(client, ChatPath(request.Model),
            NativeOptionsPayload.Build(request, Id, Json, false), ChatAuth(), headers, Json, ct);
    }

    public override async IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        PrepareCompatibleRequest(request);
        await foreach (var part in NativeHttp.StreamAsync<Dictionary<string, JsonElement>, ChatCompletionUpdate>(client, ChatPath(request.Model),
            NativeOptionsPayload.Build(request, Id, Json, true), ChatAuth(), headers, Json,
            part => part.AdditionalProperties?.ContainsKey("error") == true, ct)) yield return part;
    }
}
