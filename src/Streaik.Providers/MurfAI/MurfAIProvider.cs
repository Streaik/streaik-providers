using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Streaik.ChatCompletions.Models;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class MurfAIProviderRegistration
{
    public static IServiceCollection AddMurfAI(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, MurfAIProvider.Id, "https://api.murf.ai/v1/");
        return services.AddProvider<MurfAIProvider>(MurfAIProvider.Id);
    }
}

internal sealed partial class MurfAIProvider(IHttpClientFactory clients, IProviderCredentials credentials, IConfiguration configuration)
    : IModelListingProvider, IChatCompletionsProvider, INativeChatOptionsProvider, IChatAudioRoutingProvider
{
    internal const string Id = "murfai";
    private const string Falcon = "falcon-2.1";
    private static readonly string[] Versions = ["gen2", Falcon];
    private static readonly (string Locale, string Name)[] Languages =
    [
        ("en-US", "English (US & Canada)"), ("en-UK", "English (UK)"), ("en-IN", "English (India)"),
        ("en-AU", "English (Australia)"), ("en-SCOTT", "English (Scotland)"), ("es-MX", "Spanish (Mexico)"),
        ("es-ES", "Spanish (Spain)"), ("fr-FR", "French (France)"), ("de-DE", "German (Germany)"),
        ("it-IT", "Italian (Italy)"), ("nl-NL", "Dutch (Netherlands)"), ("pt-BR", "Portuguese (Brazil)"),
        ("zh-CN", "Chinese (Mandarin, China)"), ("ja-JP", "Japanese (Japan)"), ("ko-KR", "Korean (Korea)"),
        ("hi-IN", "Hindi (India)"), ("ta-IN", "Tamil (India)"), ("bn-IN", "Bangla (India)"),
        ("hr-HR", "Croatian (Croatia)"), ("sk-SK", "Slovak (Slovakia)"), ("pl-PL", "Polish (Poland)"), ("el-GR", "Greek (Greece)")
    ];
    private static readonly HashSet<string> TranslationControls = new(StringComparer.OrdinalIgnoreCase)
    {
        "model", "input", "messages", "instructions", "texts", "stream", "stream_options",
        "metadata", "store", "tools", "tool_choice", "temperature", "max_tokens", "max_completion_tokens", "top_p",
        "response_format", "reasoning_effort", "background", "conversation", "previous_response_id"
    };

    private async Task<HttpResponseMessage> SendAsync(HttpClient client, string path, JsonObject? payload, CancellationToken ct)
    {
        using var request = new HttpRequestMessage(payload is null ? HttpMethod.Get : HttpMethod.Post, path);
        if (payload is not null) request.Content = JsonContent.Create(payload, options: Json);
        request.Headers.Add("api-key", RequiredKey(credentials, Id));
        return await NativeHttp.SendAsync(client, request, ct);
    }

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(credentials.GetApiKey(Id))) yield break;
        using var client = clients.CreateClient(Id);
        foreach (var version in Versions)
        {
            yield return new() { Id = version, Name = version, Type = "speech", OwnedBy = Id };
            using var response = await SendAsync(client, $"speech/voices?model={version}", null, ct);
            var voices = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var voice in Array(voices))
            {
                if (String(voice, "voiceId")?.Trim() is not { Length: > 0 } id || !seen.Add(id)) continue;
                var locales = Property(voice, "supportedLocales");
                var tags = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "voice" };
                if (String(voice, "locale") is { Length: > 0 } locale) tags.Add(locale.Split('-')[0]);
                if (locales.ValueKind == JsonValueKind.Object)
                    foreach (var entry in locales.EnumerateObject()) tags.Add(entry.Name.Split('-')[0]);
                yield return new() { Id = $"{version}/{id}", Name = $"{String(voice, "displayName") ?? id} ({version})",
                    Type = "speech", OwnedBy = Id, Description = String(voice, "description"), Tags = tags };
            }
        }
        foreach (var (locale, name) in Languages)
            yield return new() { Id = $"translate/{locale}", Name = name, Type = "language", OwnedBy = Id,
                Description = $"Translate to {name}.", Tags = ["translate", locale.Split('-')[0]] };
    }

    public bool IsSpeechModel(string model) => NormalizeVersion(model.Split('/')[0]) is "gen2" or Falcon;
    public bool IsTranscriptionModel(string model) => false;

    public async Task<ChatCompletion> CompleteChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var (texts, metadata) = await TranslateAsync(request, ct);
        return new() { Id = $"chatcmpl-{Guid.NewGuid():N}", Model = request.Model, Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            Choices = [new() { Index = 0, FinishReason = "stop", Message = new() { Role = "assistant", Content = string.Join('\n', texts) } }],
            AdditionalProperties = metadata };
    }

    public async IAsyncEnumerable<ChatCompletionUpdate> StreamChatAsync(ChatCompletionOptions request, IReadOnlyDictionary<string, string> headers,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var (texts, metadata) = await TranslateAsync(request, ct);
        var id = $"chatcmpl-{Guid.NewGuid():N}";
        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        yield return Update(new() { Role = "assistant" });
        for (var i = 0; i < texts.Length; i++) yield return Update(new() { Content = (i == 0 ? "" : "\n") + texts[i] });
        var done = Update(new(), "stop");
        done.AdditionalProperties = metadata;
        yield return done;

        ChatCompletionUpdate Update(ChatCompletionDelta delta, string? finish = null)
            => new() { Id = id, Model = request.Model, Created = created, Choices = [new() { Index = 0, Delta = delta, FinishReason = finish }] };
    }

    private async Task<(string[] Texts, Dictionary<string, JsonElement> Metadata)> TranslateAsync(ChatCompletionOptions request, CancellationToken ct)
    {
        var model = request.Model.Split('/', 2);
        if (!model[0].Equals("translate", StringComparison.OrdinalIgnoreCase)) throw new ProviderRequestException("Murf AI text requests require a translate model.", 422);
        var options = new Dictionary<string, JsonElement>(NativeMediaOptions.Read(Id, request.AdditionalProperties, TranslationControls), StringComparer.OrdinalIgnoreCase);
        var language = model.Length == 2 ? model[1] : null;
        if (string.IsNullOrWhiteSpace(language)) language = OptionString(options, "targetLanguage") ?? OptionString(options, "language");
        options.Remove("language");
        options.Remove("targetLanguage");
        if (string.IsNullOrWhiteSpace(language)) throw new ProviderRequestException("Murf AI translation requires a target language.");

        var texts = request.Messages.Where(m => m.Role == "user").SelectMany(m => TextParts(JsonSerializer.SerializeToElement(m.Content, Json)))
            .Where(t => !string.IsNullOrWhiteSpace(t)).ToArray();
        if (texts.Length == 0) throw new ProviderRequestException("Murf AI translation requires user text.");
        var payload = ToObject(options);
        payload["targetLanguage"] = language.Trim();
        payload["texts"] = JsonSerializer.SerializeToNode(texts, Json);
        using var client = clients.CreateClient(Id);
        using var response = await SendAsync(client, "text/translate", payload, ct);
        var raw = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        var translated = Array(Property(raw, "translations")).Select(item => String(item, "translated_text")).ToArray();
        if (translated.Length != texts.Length || translated.Any(t => t is null)) throw new JsonException("Invalid Murf AI translation result.");
        return (translated.Select(t => t!).ToArray(), new() { ["providerMetadata"] = JsonSerializer.SerializeToElement(NativeMediaOptions.Metadata(Id, raw), Json) });
    }

    private static IEnumerable<string> TextParts(JsonElement content)
    {
        if (content.ValueKind == JsonValueKind.String) yield return content.GetString()!;
        else foreach (var part in Array(content))
            if (String(part, "type") == "text" && String(part, "text") is { } text) yield return text;
    }

    private static string? OptionString(IReadOnlyDictionary<string, JsonElement> values, string key)
        => values.TryGetValue(key, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
}
