using System.Globalization;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Streaik.OpenAI.Models;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

public static class CAMBAIProviderRegistration
{
    public static IServiceCollection AddCAMBAI(this IServiceCollection services)
    {
        const string address = "https://client.camb.ai/apis/";
        ProviderServiceCollectionExtensions.AddClient(services, CambaiProvider.Id, address);
        ProviderServiceCollectionExtensions.AddClient(services, CambaiProvider.AssetClient, address);
        return services.AddProvider<CambaiProvider>(CambaiProvider.Id);
    }
}

internal sealed partial class CambaiProvider(IHttpClientFactory clients, IProviderCredentials credentials, IConfiguration configuration)
    : IModelListingProvider
{
    internal const string Id = "cambai";
    internal const string AssetClient = "cambai-assets";
    private const string Source = "/source-language/";
    private const string Target = "/target-language/";

    public async IAsyncEnumerable<Model> ListModelsAsync([EnumeratorCancellation] CancellationToken ct)
    {
        if (credentials.GetApiKey(Id) is not { Length: > 0 } key || string.IsNullOrWhiteSpace(key)) yield break;
        yield return Entry("text-to-sound/sound", "Text-to-sound effects", "speech");
        yield return Entry("text-to-sound/music", "Text-to-music", "speech");
        yield return Entry("text-to-voice", "Text-to-voice preview", "speech");

        using var client = clients.CreateClient(Id);
        var sources = await Languages("source-languages");
        foreach (var (source, name) in sources)
        {
            yield return Entry($"tts{Source}{source}", $"Text-to-Speech ({name})", "speech");
            yield return Entry($"transcribe{Source}{source}", $"Transcribe ({name})", "transcription");
        }
        var targets = await Languages("target-languages");
        foreach (var (source, from) in sources)
            foreach (var (target, to) in targets)
            {
                ct.ThrowIfCancellationRequested();
                if (source == target) continue;
                var pair = $"{Source}{source}{Target}{target}";
                yield return Entry($"translate{pair}", $"Translate {from} to {to}", "language", ["translate"]);
                yield return Entry($"translated-tts{pair}", $"Translated TTS {from} to {to}", "speech");
            }

        async Task<Dictionary<int, string>> Languages(string path)
        {
            var body = await SendJsonAsync(client, key, path, null, ct);
            if (body.ValueKind != JsonValueKind.Array) throw new JsonException("Expected a CAMBAI language array.");
            var languages = new Dictionary<int, string>();
            foreach (var item in body.EnumerateArray())
                if (PositiveNumber(Property(item, "id")) is { } id)
                    languages.TryAdd(id, String(item, "language") ?? String(item, "shortName") ?? Number(id));
            return languages;
        }

        static Model Entry(string model, string name, string type, string[]? tags = null)
            => new() { Id = model, Name = name, OwnedBy = "CAMB.AI", Type = type, Tags = tags };
    }

    private static (int Source, int? Target) LanguageIds(string model, string operation, bool pair = false)
    {
        var prefix = operation + Source;
        var parts = model.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)
            ? model[prefix.Length..].Split(Target, StringSplitOptions.None) : [];
        if (parts.Length != (pair ? 2 : 1) || !Parse(parts[0], out var source))
            throw new ProviderRequestException($"Invalid CAMBAI {operation} model.");
        if (!pair) return (source, null);
        if (!Parse(parts[1], out var target) || source == target)
            throw new ProviderRequestException($"Invalid CAMBAI {operation} language pair.");
        return (source, target);

        static bool Parse(string value, out int id)
            => int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out id) && id > 0;
    }

    private static string Number(int value) => value.ToString(CultureInfo.InvariantCulture);

    private static int? PositiveNumber(JsonElement value)
        => value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out var id) && id > 0 ? id : null;

    private async Task<Job> StartJobAsync(string operation, HttpContent payload, CancellationToken ct)
    {
        var key = RequiredKey(credentials, Id);
        using var client = clients.CreateClient(Id);
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromMinutes(10));
        var token = deadline.Token;
        var create = await SendJsonAsync(client, key, operation, payload, token);
        if (String(create, "task_id") is not { Length: > 0 and <= 256 } id || string.IsNullOrWhiteSpace(id) || id is "." or "..")
            throw new JsonException("Missing or invalid CAMBAI task ID.");

        var statusPath = $"{operation}/{Uri.EscapeDataString(id)}";
        while (true)
        {
            var status = await SendJsonAsync(client, key, statusPath, null, token);
            switch (String(status, "status")?.ToUpperInvariant())
            {
                case "SUCCESS":
                    return new(id, PositiveNumber(Property(status, "run_id")) ?? throw new JsonException("Missing CAMBAI run ID."), create, status);
                case "ERROR" or "FAILED" or "TIMEOUT" or "PAYMENT_REQUIRED" or "CANCELLED" or "CANCELED":
                    throw new HttpRequestException("CAMBAI job failed.");
                case null or "":
                    throw new JsonException("Missing CAMBAI job status.");
                default:
                    await Task.Delay(TimeSpan.FromSeconds(operation == "transcribe" ? 5 : 2), token);
                    break;
            }
        }
    }

    private async Task<JsonElement> JobResultAsync(string resource, Job job, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        return await SendJsonAsync(client, RequiredKey(credentials, Id), $"{resource}/{Number(job.RunId)}", null, ct);
    }

    private static async Task<JsonElement> SendJsonAsync(HttpClient client, string key, string path, HttpContent? payload, CancellationToken ct)
    {
        using var request = new HttpRequestMessage(payload is null ? HttpMethod.Get : HttpMethod.Post, path) { Content = payload };
        request.Headers.Add("x-api-key", key);
        request.Headers.Accept.ParseAdd("application/json");
        using var response = await NativeHttp.SendAsync(client, request, ct);
        return await NativeHttp.ReadAsync<JsonElement>(response, Json, ct);
    }

    private static JsonContent JobPayload(Dictionary<string, JsonElement> native, params (string Name, object Value)[] fields)
    {
        foreach (var (name, value) in fields) native[name] = JsonSerializer.SerializeToElement(value, Json);
        return JsonContent.Create(native, options: Json);
    }

    private sealed record Job(string TaskId, int RunId, JsonElement Create, JsonElement Status)
    {
        public object Raw(object result) => new { task_id = TaskId, run_id = RunId, create = Create, status = Status, result };
    }
}
