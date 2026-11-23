using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Streaik.Providers.OpenAI;

public sealed partial class OpenAIProvider
{
    public IAsyncEnumerable<ProviderSkill> ListSkillsAsync(CancellationToken ct)
        => ReadSkillPages<ProviderSkill>("v1/skills", ct);

    public IAsyncEnumerable<ProviderSkillVersion> ListSkillVersionsAsync(string skillId, CancellationToken ct)
        => ReadSkillPages<ProviderSkillVersion>($"v1/skills/{SkillSegment(skillId)}/versions", ct);

    public async Task<byte[]> DownloadSkillAsync(string skillId, string? version, CancellationToken ct)
    {
        var path = $"v1/skills/{SkillSegment(skillId)}";
        if (version is not null) path += $"/versions/{SkillSegment(version)}";
        using var client = clients.CreateClient(Id);
        using var request = new HttpRequestMessage(HttpMethod.Get, path + "/content");
        Authenticate(request.Headers, RequireKey());
        using var response = await NativeHttp.SendAsync(client, request, ct);
        if (response.Content.Headers.ContentLength > SkillArchiveLimits.MaxBytes)
            throw new ProviderRequestException("The provider skill ZIP exceeds the size limit.", 413);
        await using var source = await response.Content.ReadAsStreamAsync(ct);
        using var output = new MemoryStream();
        var buffer = new byte[65536];
        for (int read; (read = await source.ReadAsync(buffer, ct)) != 0;)
        {
            if (output.Length + read > SkillArchiveLimits.MaxBytes)
                throw new ProviderRequestException("The provider skill ZIP exceeds the size limit.", 413);
            output.Write(buffer, 0, read);
        }
        return output.ToArray();
    }

    public async Task<ProviderSkill> UploadSkillAsync(byte[] archive, CancellationToken ct)
    {
        using var client = clients.CreateClient(Id);
        using var request = new HttpRequestMessage(HttpMethod.Post, "v1/skills");
        Authenticate(request.Headers, RequireKey());
        using var multipart = new MultipartFormDataContent();
        var file = new ByteArrayContent(archive);
        file.Headers.ContentType = new MediaTypeHeaderValue("application/zip");
        multipart.Add(file, "files", "skill.zip");
        request.Content = multipart;
        using var response = await NativeHttp.SendAsync(client, request, ct);
        return await NativeHttp.ReadAsync<ProviderSkill>(response, JsonSerializerOptions.Web, ct);
    }

    private async IAsyncEnumerable<T> ReadSkillPages<T>(string path, [EnumeratorCancellation] CancellationToken ct)
    {
        var key = RequireKey();
        using var client = clients.CreateClient(Id);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        string? after = null;
        for (var i = 0; i < 100; i++)
        {
            var url = after is null ? path : $"{path}?after={Uri.EscapeDataString(after)}";
            var page = await NativeHttp.GetAsync<SkillPage<T>>(client, url, headers => Authenticate(headers, key), ct);
            foreach (var item in page.Data ?? []) yield return item;
            if (!page.HasMore) yield break;
            after = page.LastId;
            if (string.IsNullOrWhiteSpace(after) || !seen.Add(after)) break;
        }
        throw new ProviderRequestException("The provider returned an incomplete skill catalog.", 502);
    }

    private static string SkillSegment(string value)
        => !string.IsNullOrWhiteSpace(value) && value is not ("." or "..")
            ? Uri.EscapeDataString(value) : throw new ArgumentException("Invalid skill path segment.", nameof(value));

    private sealed record SkillPage<T>(T[]? Data,
        [property: JsonPropertyName("has_more")] bool HasMore,
        [property: JsonPropertyName("last_id")] string? LastId);
}
