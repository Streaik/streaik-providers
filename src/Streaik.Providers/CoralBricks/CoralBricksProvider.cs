using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.OpenAI.Models;
using Streaik.Responses;
using Streaik.Responses.Streaming;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class CoralBricksRegistration
{
    public static IServiceCollection AddCoralBricks(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, CoralBricksProvider.Id, "https://inference.coralbricks.ai/v1/");
        return services.AddProvider<CoralBricksProvider>(CoralBricksProvider.Id);
    }
}

internal sealed class CoralBricksProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), IResponsesProvider, INativeChatOptionsProvider
{
    internal const string Id = "coralbricks";
    private static readonly HashSet<string> EnvelopeFields = new(StringComparer.OrdinalIgnoreCase)
        { Id, "model", "stream", "store", "background", "previous_response_id", "conversation", "input", "messages" };

    public override async Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        return await NativeHttp.PostAsync<Dictionary<string, JsonElement>, ChatCompletion>(client, "chat/completions",
            Payload(request, false, Json), Bearer(ApiKey()), headers, Json, ct);
    }

    public override async IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        await foreach (var part in NativeHttp.StreamAsync<Dictionary<string, JsonElement>, ChatCompletionUpdate>(client,
            "chat/completions", Payload(request, true, Json), Bearer(ApiKey()), headers, Json,
            part => part.AdditionalProperties?.ContainsKey("error") == true, ct)) yield return part;
    }

    public async Task<ResponseResult> CreateResponseAsync(ResponseRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        using var client = CreateClient();
        return await NativeHttp.PostAsync<Dictionary<string, JsonElement>, ResponseResult>(client, "responses",
            Payload(request, false, ResponseJson.Default), Bearer(ApiKey()), headers, ResponseJson.Default, ct);
    }

    public async IAsyncEnumerable<ResponseStreamPart> StreamResponseAsync(ResponseRequest request,
        IReadOnlyDictionary<string, string> headers, [EnumeratorCancellation] CancellationToken ct)
    {
        using var client = CreateClient();
        await foreach (var part in NativeHttp.StreamAsync<Dictionary<string, JsonElement>, ResponseStreamPart>(client,
            "responses", Payload(request, true, ResponseJson.Default), Bearer(ApiKey()), headers, ResponseJson.Default,
            part => part.Type is "response.completed" or "response.failed" or "response.incomplete" or "error", ct)) yield return part;
    }

    private static Dictionary<string, JsonElement> Payload<T>(T request, bool stream, JsonSerializerOptions json)
    {
        var source = JsonSerializer.SerializeToElement(request, json);
        var body = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (var property in source.EnumerateObject())
            if (property.Name != Id) body[property.Name] = property.Value;

        // Preserve native values verbatim, but leave routing, history, and state controls to the typed request.
        // In-process callers can set typed store/background flags; the gateway remains stateless.
        if (source.TryGetProperty(Id, out var options))
        {
            if (options.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("CoralBricks options must be an object.");
            foreach (var property in options.EnumerateObject())
                if (!EnvelopeFields.Contains(property.Name)) body[property.Name] = property.Value;
        }
        body["stream"] = JsonSerializer.SerializeToElement(stream);
        return body;
    }

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (ConfiguredApiKey() is not { Length: > 0 } key || string.IsNullOrWhiteSpace(key)) yield break;
        var result = await FetchModelsAsync(key, ct);
        var data = result.ValueKind == JsonValueKind.Array ? result : Property(result, "data");
        if (data.ValueKind != JsonValueKind.Array) throw new JsonException("CoralBricks model catalog must contain a data array.");
        foreach (var model in data.EnumerateArray())
        {
            if (String(model, "id") is not { } id || string.IsNullOrWhiteSpace(id)) continue;
            var tags = new HashSet<string>(StringComparer.Ordinal);
            foreach (var name in new[] { "tags", "capabilities", "supported_parameters" })
            {
                var value = Property(model, name);
                foreach (var tag in Array(value))
                    if (tag.ValueKind == JsonValueKind.String) tags.Add(tag.GetString()!);
                if (value.ValueKind == JsonValueKind.Object)
                    foreach (var capability in value.EnumerateObject())
                        if (capability.Value.ValueKind == JsonValueKind.True) tags.Add(capability.Name);
            }
            var created = Property(model, "created");
            yield return new Model { Id = id, Name = String(model, "name") ?? id, Type = String(model, "type") ?? "language",
                OwnedBy = String(model, "owned_by") ?? "CoralBricks", Description = String(model, "description"),
                Created = created.ValueKind == JsonValueKind.Number && created.TryGetInt64(out var timestamp) ? timestamp : null,
                ContextWindow = Number(model, "context_window", "context_length", "max_model_len"),
                MaxTokens = Number(model, "max_tokens", "max_completion_tokens", "max_output_tokens"),
                Tags = tags.Count > 0 ? tags : null };
        }
    }

    private static int? Number(JsonElement model, params string[] names)
    {
        foreach (var name in names)
            if (Property(model, name) is { ValueKind: JsonValueKind.Number } value && value.TryGetInt32(out var number)) return number;
        return null;
    }
}
