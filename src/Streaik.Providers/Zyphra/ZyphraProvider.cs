using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Zyphra;

public static class ZyphraRegistration
{
    public static IServiceCollection AddZyphra(this IServiceCollection services)
    {
        ProviderServiceCollectionExtensions.AddClient(services, ZyphraProvider.Id, "https://api.zyphracloud.com/api/v1/");
        return services.AddProvider<ZyphraProvider>(ZyphraProvider.Id);
    }
}

internal sealed class ZyphraProvider(IHttpClientFactory clients, IProviderCredentials credentials)
    : CompatibleChatProvider(clients, credentials, Id), IStreamingSpeechProvider, INativeChatOptionsProvider
{
    internal const string Id = "zyphra";
    private const string SpeechModel = "zyphra/ZONOS2";

    public override async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        (string Id, string Name)[] languageModels =
        [
            ("deepseek-ai/DeepSeek-V3.2", "DeepSeek-V3.2"), ("moonshotai/Kimi-K2.6", "Kimi-K2.6"),
            ("XiaomiMiMo/MiMo-V2.5-Pro", "MiMo-V2.5-Pro"), ("zai-org/GLM-5.1-FP8", "GLM-5.1-FP8"), ("zyphra/ZAYA1-8B", "ZAYA1-8B")
        ];
        foreach (var (id, name) in languageModels) yield return new() { Id = id, Name = name, Type = "language", OwnedBy = "Zyphra" };
        yield return new() { Id = SpeechModel, Name = "ZONOS2", Type = "speech", OwnedBy = "Zyphra" };

        if (ConfiguredApiKey() is not { } key || string.IsNullOrWhiteSpace(key)) yield break;
        using var client = CreateClient();
        Action<HttpRequestHeaders> auth = headers => headers.Authorization = new("Bearer", key);
        var defaults = NativeHttp.GetAsync<JsonElement>(client, "audio/default-voices", auth, ct);
        var clones = NativeHttp.GetAsync<JsonElement>(client, "audio/cloned-voices", auth, ct);
        await Task.WhenAll(defaults, clones);
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var (kind, voices) in new[] { ("default", defaults.Result), ("cloned", clones.Result) })
        {
            foreach (var voice in Array(voices))
            {
                if (String(voice, "voice_id")?.Trim() is not { Length: > 0 } id || !seen.Add(id)) continue;
                var name = String(voice, "display_name") ?? String(voice, "name") ?? id;
                var description = $"Zyphra ZONOS2 {kind} voice shortcut for {name} ({id}).";
                if (String(voice, "description") is { Length: > 0 } extra) description += " " + extra;
                yield return new() { Id = $"{SpeechModel}/{id}", Name = $"ZONOS2 {name} ({id})", Type = "speech",
                    OwnedBy = "Zyphra", Description = description, Tags = ["voice"] };
            }
        }
    }

    public Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeakAsync(request, headers, stream: false, ct);

    public Task<HttpResponseMessage> SpeakStreamingAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, CancellationToken ct)
        => SpeakAsync(request, headers, stream: true, ct);

    private async Task<HttpResponseMessage> SpeakAsync(AudioSpeechRequest request, IReadOnlyDictionary<string, string> headers, bool stream, CancellationToken ct)
    {
        var model = request.Model.Trim();
        string? shortcut = null;
        if (!model.Equals(SpeechModel, StringComparison.OrdinalIgnoreCase))
        {
            if (!model.StartsWith(SpeechModel + "/", StringComparison.OrdinalIgnoreCase)
                || (shortcut = model[(SpeechModel.Length + 1)..].Trim()).Length == 0)
                throw new ProviderRequestException($"Zyphra speech requires '{SpeechModel}' or '{SpeechModel}/<voice_id>'.");
        }

        var payload = new JsonObject { ["model"] = SpeechModel, ["input"] = request.Input, ["stream"] = stream };
        string? providerVoice = null;
        foreach (var (name, value) in request.AdditionalProperties ?? [])
        {
            if (name.Equals("voice", StringComparison.Ordinal) && value.ValueKind == JsonValueKind.String) providerVoice = value.GetString();
            if (name.ToLowerInvariant() is "model" or "input" or "stream" or "stream_format" or "voice" or "instructions" or "response_format" or "speed") continue;
            payload[name] = Copy(value);
        }
        var voice = shortcut ?? (string.IsNullOrWhiteSpace(request.Voice) ? providerVoice : request.Voice);
        if (!string.IsNullOrWhiteSpace(voice)) payload["voice"] = voice.Trim();
        if (request.Speed is { } speed) payload["speed"] = speed;
        var format = request.ResponseFormat?.Trim().ToLowerInvariant() switch
        {
            "mp3" or "mpeg" or "audio/mpeg" or "audio/mp3" => "mp3",
            "opus" or "audio/opus" or "audio/ogg" => "opus",
            "wav" or "audio/wav" or "audio/x-wav" => "wav",
            "flac" or "audio/flac" or "audio/x-flac" => "flac",
            "aac" or "audio/aac" => "aac",
            "pcm" or "audio/pcm" or "application/octet-stream" => "pcm",
            "m4a" or "mp4" or "audio/mp4" or "audio/m4a" or "audio/x-m4a" => "m4a",
            _ => null
        };
        if (format is not null) payload["response_format"] = format;

        using var client = CreateClient();
        using var message = new HttpRequestMessage(HttpMethod.Post, "audio/speech") { Content = JsonContent.Create(payload, options: Json) };
        message.Headers.Authorization = new("Bearer", ApiKey());
        message.Headers.Accept.ParseAdd("audio/*");
        NativeHttp.ApplyHeaders(message, headers);
        var response = await NativeHttp.SendAsync(client, message, ct);
        response.Content.Headers.ContentType ??= new(format switch
        {
            "m4a" => "audio/mp4", "pcm" => "application/octet-stream", null or "mp3" => "audio/mpeg", _ => "audio/" + format
        });
        return response;
    }
}
