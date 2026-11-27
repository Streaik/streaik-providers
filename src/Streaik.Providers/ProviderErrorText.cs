using System.Net;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Streaik.Providers;

// The exception message stays generic for callers and logs. Only the gateway
// deliberately exposes the separately validated diagnostic in its error envelope.
internal sealed class UpstreamHttpException(HttpStatusCode status, string diagnostic)
    : HttpRequestException("The upstream provider rejected the request.", null, status)
{
    internal string Diagnostic { get; } = diagnostic;
}

internal static partial class ProviderErrorText
{
    private const int MaxBodyBytes = 16 * 1024;
    private const int MaxMessageLength = 2048;

    internal static async Task<string?> ReadAsync(HttpResponseMessage response, CancellationToken ct)
    {
        var content = response.Content;
        var mediaType = content.Headers.ContentType?.MediaType;
        if (content.Headers.ContentLength > MaxBodyBytes || mediaType is null
            || !(mediaType.Equals("application/json", StringComparison.OrdinalIgnoreCase)
                || mediaType.EndsWith("+json", StringComparison.OrdinalIgnoreCase))) return null;

        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromSeconds(2));
        try
        {
            await using var stream = await content.ReadAsStreamAsync(deadline.Token);
            var bytes = new byte[MaxBodyBytes + 1];
            var count = await stream.ReadAtLeastAsync(bytes, bytes.Length, throwOnEndOfStream: false, deadline.Token);
            if (count > MaxBodyBytes) return null;
            using var document = JsonDocument.Parse(bytes.AsMemory(0, count));
            return FromJson(document.RootElement);
        }
        catch (Exception error) when (!ct.IsCancellationRequested && error is IOException or HttpRequestException or JsonException or OperationCanceledException)
        {
            return null; // Do not replace an upstream HTTP status with a body-read error.
        }
    }

    internal static string? FromJson(JsonElement root)
    {
        var detail = Property(root, "detail");
        return Message(Property(root, "error")) ?? Message(root) ?? Message(detail) ?? ReadString(detail);
    }

    internal static string? Message(JsonElement error) => ReadString(Property(error, "message"));
    private static JsonElement Property(JsonElement value, string name) =>
        value.ValueKind == JsonValueKind.Object && value.TryGetProperty(name, out var property) ? property : default;

    private static string? ReadString(JsonElement value) => value.ValueKind == JsonValueKind.String
        && value.GetString() is { } text && !string.IsNullOrWhiteSpace(text) ? text : null;

    internal static bool IsCredentialHeader(string name) => name.Equals("Authorization", StringComparison.OrdinalIgnoreCase)
        || name.Equals("Proxy-Authorization", StringComparison.OrdinalIgnoreCase)
        || name.Equals("Cookie", StringComparison.OrdinalIgnoreCase)
        || name.EndsWith("-key", StringComparison.OrdinalIgnoreCase) || name.EndsWith("-token", StringComparison.OrdinalIgnoreCase)
        || name.EndsWith("-secret", StringComparison.OrdinalIgnoreCase) || name.EndsWith("-authorization", StringComparison.OrdinalIgnoreCase)
        || name.EndsWith("-cookie", StringComparison.OrdinalIgnoreCase) || name.Equals("api_key", StringComparison.OrdinalIgnoreCase);

    internal static IEnumerable<string> CredentialValues(string header)
    {
        yield return header;
        // Authorization includes a scheme; providers often echo just the value.
        var separator = header.IndexOf(' ');
        if (separator > 0) yield return header[(separator + 1)..].Trim();
    }

    internal static string? Sanitize(string? message, IEnumerable<string> credentials)
    {
        if (string.IsNullOrWhiteSpace(message) || message.Length > MaxMessageLength
            || message.Any(c => char.IsControl(c) && c is not ('\r' or '\n' or '\t'))
            || TraceOrMarkup().IsMatch(message)) return null;

        foreach (var value in credentials.Distinct(StringComparer.Ordinal).OrderByDescending(value => value.Length))
        {
            if (string.IsNullOrWhiteSpace(value)) continue;
            // Do not distort diagnostics for a short key that occurs inside a word.
            if (value.Length < 8)
                message = Regex.Replace(message, $@"(?<![\w-]){Regex.Escape(value)}(?![\w-])", "[redacted]", RegexOptions.CultureInvariant);
            else
                message = message.Replace(value, "[redacted]", StringComparison.Ordinal);
        }
        return Token().Replace(message, "[redacted]");
    }

    [GeneratedRegex(@"<!doctype\s|<html(?:\s|>)|stack\s*trace|traceback \(most recent call last\)|(?:\r?\n)\s*at\s+[^\r\n]+\(", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex TraceOrMarkup();

    [GeneratedRegex(@"\bsk-[A-Za-z0-9_*\-]{8,}|\bBearer\s+[A-Za-z0-9_./+\-]{12,}=*", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex Token();
}
