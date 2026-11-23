using System.Text.Json.Serialization;

namespace Streaik.Providers;

public sealed record ProviderSkill(string Id, string Name, string Description,
    [property: JsonPropertyName("created_at")] long? CreatedAt,
    [property: JsonPropertyName("default_version")] string? DefaultVersion,
    [property: JsonPropertyName("latest_version")] string? LatestVersion)
{
    public string Object => "skill";
}

public sealed record ProviderSkillVersion(string Id, string Name, string Description,
    [property: JsonPropertyName("created_at")] long? CreatedAt,
    [property: JsonPropertyName("skill_id")] string SkillId, string Version)
{
    public string Object => "skill.version";
}

public interface ISkillReadProvider
{
    IAsyncEnumerable<ProviderSkill> ListSkillsAsync(CancellationToken ct);
    IAsyncEnumerable<ProviderSkillVersion> ListSkillVersionsAsync(string skillId, CancellationToken ct);
    Task<byte[]> DownloadSkillAsync(string skillId, string? version, CancellationToken ct);
}

public interface ISkillUploadProvider
{
    Task<ProviderSkill> UploadSkillAsync(byte[] archive, CancellationToken ct);
}

public static class SkillArchiveLimits
{
    public const int MaxBytes = 16 * 1024 * 1024;
    public const int MaxFiles = 512;
}
