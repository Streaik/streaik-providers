using System.Runtime.CompilerServices;
using System.Text.Json;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Mistral;

internal sealed partial class MistralProvider
{
    public bool IsSpeechModel(string model)
    {
        var name = model.AsSpan();
        var slash = name.IndexOf('/');
        if (slash >= 0) name = name[..slash];
        return name.StartsWith("voxtral-", StringComparison.Ordinal) && name.Contains("-tts", StringComparison.Ordinal);
    }

    public bool IsTranscriptionModel(string model) => !IsRealtimeTranscription(model)
        && (model.StartsWith("voxtral-mini-transcribe-", StringComparison.Ordinal) || model.StartsWith("voxtral-small-transcribe-", StringComparison.Ordinal));

    private static Dictionary<string, JsonElement> AudioOptions(IReadOnlyDictionary<string, JsonElement>? options, bool namespaced = false)
    {
        if (options?.TryGetValue(Id, out var native) == true)
        {
            if (native.ValueKind == JsonValueKind.Null) return new(StringComparer.Ordinal);
            if (native.ValueKind != JsonValueKind.Object) throw new ProviderRequestException("providerOptions.mistral must be an object.");
            return native.EnumerateObject().ToDictionary(x => x.Name, x => x.Value, StringComparer.Ordinal);
        }
        return namespaced || options is null ? new(StringComparer.Ordinal) : new(options, StringComparer.Ordinal);
    }

    private static string RequiredString(JsonElement body, string name)
        => String(body, name) ?? throw new JsonException($"Mistral audio did not return '{name}'.");

    private static double? AudioDouble(JsonElement body, string name)
        => Property(body, name) is { ValueKind: JsonValueKind.Number } value && value.TryGetDouble(out var number) && double.IsFinite(number)
            ? number : null;

    private static long? AudioCount(JsonElement body, string name)
        => Property(body, name) is { ValueKind: JsonValueKind.Number } value && value.TryGetInt64(out var count) && count >= 0 ? count : null;

    private static Dictionary<string, JsonElement> AudioMetadata(JsonElement body) => new() { [Id] = body };

    private async Task<HttpResponseMessage> SendAudioAsync(HttpClient client, HttpRequestMessage request,
        IReadOnlyDictionary<string, string> headers, string accept, CancellationToken ct)
    {
        Bearer(ApiKey())(request.Headers);
        request.Headers.Accept.ParseAdd(accept);
        NativeHttp.ApplyHeaders(request, headers);
        return await NativeHttp.SendAsync(client, request, ct);
    }

    // Audio uses named events as well as data-only frames. Do not echo raw SSE errors.
    private static async IAsyncEnumerable<(string? Type, JsonElement Payload)> AudioEventsAsync(HttpResponseMessage response,
        string done, [EnumeratorCancellation] CancellationToken ct)
    {
        if (response.Content.Headers.ContentType?.MediaType != "text/event-stream")
            throw new IOException("Expected Mistral audio event stream.");
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        using var reader = new StreamReader(stream);
        var lines = new List<string>();
        string? name = null;
        while (true)
        {
            var line = await reader.ReadLineAsync(ct);
            if (string.IsNullOrEmpty(line))
            {
                if (name == "error") throw new IOException("Mistral audio stream reported an error.");
                if (lines.Count > 0)
                {
                    var data = string.Join('\n', lines);
                    if (data.Trim() == "[DONE]") break;
                    using var document = JsonDocument.Parse(data);
                    var payload = Unwrap(document.RootElement, "type");
                    if (payload.ValueKind != JsonValueKind.Object) throw new JsonException("Invalid Mistral audio stream frame.");
                    var type = String(payload, "type") ?? name;
                    if (type == "error" || Property(payload, "error").ValueKind is not (JsonValueKind.Undefined or JsonValueKind.Null))
                        throw new IOException("Mistral audio stream reported an error.");
                    if (name is not null && type != name) throw new JsonException("Mistral audio event does not match its payload.");
                    yield return (type, payload.Clone());
                    if (type == done) yield break;
                    lines.Clear();
                }
                name = null;
                if (line is null) break;
            }
            else if (line.StartsWith("data:", StringComparison.Ordinal)) lines.Add(line.StartsWith("data: ", StringComparison.Ordinal) ? line[6..] : line[5..]);
            else if (line.StartsWith("event:", StringComparison.Ordinal)) name = line[6..].Trim();
        }
        throw new IOException($"Mistral audio stream ended before '{done}'.");
    }

    private static byte[] AudioBytes(string base64, bool upstream)
    {
        try
        {
            if (Convert.FromBase64String(base64) is { Length: > 0 } bytes) return bytes;
        }
        catch (FormatException) { }
        if (upstream) throw new JsonException("Mistral returned invalid base64 audio.");
        throw new ProviderRequestException("Audio must contain nonempty base64 data.");
    }
}
