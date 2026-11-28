using System.Net.Http.Json;
using System.Text.Json;
using Streaik.Vercel.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Mistral;

internal sealed partial class MistralProvider
{
    private static bool IsRealtimeTranscription(string model)
        => model is "voxtral-mini-transcribe-realtime-2602" or "voxtral-mini-transcribe-realtime-3" or "voxtral-mini-transcribe-realtime-3-0";

    private static bool IsRealtimeSpeech(string model) => model == "voxtral-mini-tts-3-0";

    public async Task<RealtimeTokenResponse> CreateRealtimeTokenAsync(string model, RealtimeTokenRequest request,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        if (!IsRealtimeTranscription(model) && !IsRealtimeSpeech(model))
            throw new ProviderRequestException("Select a supported Mistral realtime speech or transcription model.");
        if (request.SessionConfig is { } config && (config.Instructions is not null || config.Voice is not null || config.OutputModalities is not null
            || config.InputAudioFormat is not null || config.OutputAudioFormat is not null || config.InputAudioTranscription is not null
            || config.OutputAudioTranscription is not null || config.HasTurnDetection || config.Tools is not null))
            throw new ProviderRequestException("Send Mistral session settings over the WebSocket before input starts.");
        var options = AudioOptions(request.SessionConfig?.ProviderOptions, namespaced: true);
        var body = new Dictionary<string, object> { ["purpose"] = "realtime", ["model"] = model };
        if (options.TryGetValue("ttl_seconds", out var ttl) && ttl.ValueKind != JsonValueKind.Null)
        {
            if (ttl.ValueKind != JsonValueKind.Number || !ttl.TryGetInt32(out var seconds) || seconds <= 0)
                throw new ProviderRequestException("Mistral ttl_seconds must be a positive integer.");
            body["ttl_seconds"] = seconds;
        }

        var key = ApiKey();
        using var client = CreateClient();
        using var message = new HttpRequestMessage(HttpMethod.Post, "client/sessions") { Content = JsonContent.Create(body) };
        Bearer(key)(message.Headers);
        NativeHttp.ApplyHeaders(message, headers);
        using var response = await client.SendAsync(message, HttpCompletionOption.ResponseHeadersRead, ct);
        // This endpoint can return secrets. Never emit diagnostics containing its response body.
        response.EnsureSuccessStatusCode();
        var result = await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
        var secret = Property(result, "client_secret");
        var token = String(secret, "value");
        var expiry = Property(secret, "expires_at");
        if (expiry.ValueKind == JsonValueKind.Undefined) expiry = Property(result, "expires_at");
        if (token is not { Length: > 3 } || !token.StartsWith("rt_", StringComparison.Ordinal) || token == key
            || token.Any(ch => !char.IsAsciiLetterOrDigit(ch) && !"!#$%&'*+-.^_`|~".Contains(ch))
            || expiry.ValueKind != JsonValueKind.String || !expiry.TryGetDateTimeOffset(out var expiresAt) || expiresAt <= DateTimeOffset.UtcNow)
            throw new IOException("Mistral returned an invalid or expired realtime client token.");
        var endpoint = IsRealtimeSpeech(model) ? "speech" : "transcriptions";
        return new() { Token = token, ExpiresAt = expiresAt.ToUnixTimeSeconds(),
            Url = $"wss://api.mistral.ai/v1/audio/{endpoint}/realtime?model={Uri.EscapeDataString(model)}" };
    }
}
