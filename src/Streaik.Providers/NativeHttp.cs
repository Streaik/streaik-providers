using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;

namespace Streaik.Providers;

internal static class NativeHttp
{
    public static async Task<T> GetAsync<T>(HttpClient client, string path, Action<HttpRequestHeaders> authenticate, CancellationToken ct)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, path);
        authenticate(request.Headers);
        using var response = await SendAsync(client, request, ct);
        return await ReadAsync<T>(response, JsonSerializerOptions.Web, ct);
    }

    public static async Task<TResult> PostAsync<TRequest, TResult>(HttpClient client, string path, TRequest body,
        Action<HttpRequestHeaders> auth, IReadOnlyDictionary<string, string> headers, JsonSerializerOptions json, CancellationToken ct)
    {
        using var request = Post(path, body, auth, headers, json, "application/json");
        using var response = await SendAsync(client, request, ct);
        return await ReadAsync<TResult>(response, json, ct);
    }

    internal static async Task<(JsonElement Body, Dictionary<string, string?> Headers)> PostJsonAsync<T>(HttpClient client, string path, T body,
        Action<HttpRequestHeaders> auth, IReadOnlyDictionary<string, string> headers, JsonSerializerOptions json, CancellationToken ct)
    {
        using var request = Post(path, body, auth, headers, json, "application/json");
        using var response = await SendAsync(client, request, ct);
        var result = await ReadAsync<JsonElement>(response, json, ct);
        var responseHeaders = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, values) in response.Headers.Concat(response.Content.Headers))
            if (!name.Equals("Set-Cookie", StringComparison.OrdinalIgnoreCase)) responseHeaders[name] = string.Join(", ", values);
        return (result, responseHeaders);
    }

    public static async IAsyncEnumerable<TEvent> StreamAsync<TRequest, TEvent>(HttpClient client, string path, TRequest body,
        Action<HttpRequestHeaders> auth, IReadOnlyDictionary<string, string> headers, JsonSerializerOptions json,
        Func<TEvent, bool> terminal, [EnumeratorCancellation] CancellationToken ct)
    {
        using var request = Post(path, body, auth, headers, json, "text/event-stream");
        using var response = await SendAsync(client, request, ct);
        await foreach (var part in ReadSseAsync(response, json, terminal, ct)) yield return part;
    }

    internal static async IAsyncEnumerable<TEvent> ReadSseAsync<TEvent>(HttpResponseMessage response, JsonSerializerOptions json,
        Func<TEvent, bool> terminal, [EnumeratorCancellation] CancellationToken ct)
    {
        if (response.Content.Headers.ContentType?.MediaType != "text/event-stream")
            throw new HttpRequestException("Expected an upstream event stream.");
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        using var reader = new StreamReader(stream);
        var data = new List<string>();
        while (await reader.ReadLineAsync(ct) is { } line)
        {
            if (line.StartsWith("data:", StringComparison.Ordinal)) { data.Add(line[5..].TrimStart(' ')); continue; }
            if (line.Length != 0 || data.Count == 0) continue;
            var payload = string.Join('\n', data);
            data.Clear();
            if (payload == "[DONE]") yield break;
            var value = JsonSerializer.Deserialize<TEvent>(payload, json) ?? throw new JsonException("Missing upstream event.");
            yield return value;
            if (terminal(value)) yield break;
        }
        throw new IOException("The upstream did not send a terminal event.");
    }

    private static HttpRequestMessage Post<T>(string path, T body, Action<HttpRequestHeaders> auth,
        IReadOnlyDictionary<string, string> headers, JsonSerializerOptions json, string accept)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, path) { Content = JsonContent.Create(body, options: json) };
        try
        {
            auth(request.Headers);
            request.Headers.Accept.ParseAdd(accept);
            ApplyHeaders(request, headers);
            return request;
        }
        catch { request.Dispose(); throw; }
    }

    internal static void ApplyHeaders(HttpRequestMessage request, IReadOnlyDictionary<string, string> headers)
    {
        foreach (var (name, value) in headers)
            if (!request.Headers.Contains(name)) request.Headers.TryAddWithoutValidation(name, value);
    }

    internal static async Task<HttpResponseMessage> SendAsync(HttpClient client, HttpRequestMessage request, CancellationToken ct)
    {
        var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
        if (response.IsSuccessStatusCode) return response;
        using (response) response.EnsureSuccessStatusCode();
        throw new HttpRequestException("Unexpected upstream response.");
    }

    internal static async Task<T> ReadAsync<T>(HttpResponseMessage response, JsonSerializerOptions json, CancellationToken ct)
        => await response.Content.ReadFromJsonAsync<T>(json, ct) ?? throw new JsonException("Missing upstream JSON.");
}
