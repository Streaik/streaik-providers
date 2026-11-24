using System.Text.Json;
using System.Text.Json.Serialization.Metadata;
using Streaik.OpenAI.Models;

namespace Streaik.Providers;

internal static class ProviderModelJson
{
    internal static readonly JsonSerializerOptions Options = new(JsonSerializerOptions.Web)
    {
        TypeInfoResolver = new DefaultJsonTypeInfoResolver { Modifiers = { IgnoreUpstreamTags } }
    };

    private static void IgnoreUpstreamTags(JsonTypeInfo type)
    {
        // The shared model contract still writes Streaik tags. Ignore the upstream field entirely,
        // including values that are not a string array, without changing the rest of the model.
        if (type.Type == typeof(Model) && type.Properties.FirstOrDefault(property => property.Name == "tags") is { } tags)
            type.Properties.Remove(tags);
    }
}
