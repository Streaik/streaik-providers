using System.Text.Json;
using System.Text.Json.Nodes;
using Streaik.Responses;
using Streaik.Responses.Streaming;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers.Mistral;

internal sealed partial class MistralProvider
{
    internal static bool IsOcrModel(string? model) => model?.StartsWith("mistral-ocr-", StringComparison.OrdinalIgnoreCase) == true;

    private static IEnumerable<OcrFile> ChatOcrFiles(JsonObject body)
        => body["messages"] is JsonArray messages
            && messages.OfType<JsonObject>().LastOrDefault(item => NodeString(item["role"]) == "user")?["content"] is JsonArray content
            ? content.OfType<JsonObject>().Select(part => NodeString(part["type"]) switch
            {
                "image_url" => new OcrFile(true, NodeString(part["image_url"] is JsonObject image ? image["url"] : part["image_url"]), null),
                "document_url" => new OcrFile(false, NodeString(part["document_url"]), null),
                "file" => new OcrFile(false, null, NodeString(part["file_id"])),
                _ => null
            }).OfType<OcrFile>() : [];

    private static IEnumerable<OcrFile> ResponseOcrFiles(JsonObject body)
        => body["input"] is JsonArray input
            && input.OfType<JsonObject>().LastOrDefault(item => NodeString(item["role"]) == "user")?["content"] is JsonArray content
            ? content.OfType<JsonObject>().Select(part => NodeString(part["type"]) switch
            {
                "input_image" => new OcrFile(true, NodeString(part["image_url"]), NodeString(part["file_id"])),
                "input_file" => new OcrFile(false, NodeString(part["file_url"]) ?? NodeString(part["file_data"]), NodeString(part["file_id"])),
                _ => null
            }).OfType<OcrFile>() : [];

    private async Task<OcrResult> ReadOcrAsync(string model, IEnumerable<OcrFile> files, JsonObject options,
        IReadOnlyDictionary<string, string> headers, CancellationToken ct)
    {
        var documents = files.Select(file => file.ToDocument()).ToArray();
        if (documents.Length == 0) throw new ProviderRequestException("Attach a document or image to the latest user message for Mistral OCR.");
        if (options["tools"] is JsonArray { Count: > 0 }) throw new ProviderRequestException("Mistral OCR does not run chat tools.");
        var markdown = new List<string>();
        var processed = 0L;
        using var client = CreateClient();
        var auth = Bearer(ApiKey());
        foreach (var document in documents)
        {
            var body = new JsonObject { ["model"] = model, ["document"] = document };
            foreach (var name in new[] { "pages", "extract_header", "extract_footer", "include_blocks", "include_image_base64",
                         "confidence_scores_granularity", "table_format", "image_limit", "image_min_size" })
                if (options[name] is { } value) body[name] = value.DeepClone();

            var response = await NativeHttp.PostAsync<JsonObject, JsonElement>(client, "ocr", body, auth, headers, Json, ct);
            if (Property(response, "pages") is not { ValueKind: JsonValueKind.Array } pages)
                throw new JsonException("Mistral OCR did not return document pages.");
            var count = Property(Property(response, "usage_info"), "pages_processed");
            processed += count.ValueKind == JsonValueKind.Number && count.TryGetInt32(out var number) && number >= 0 ? number : pages.GetArrayLength();
            foreach (var page in pages.EnumerateArray())
                foreach (var field in new[] { "header", "markdown", "footer" })
                    if (String(page, field) is { Length: > 0 } text) markdown.Add(text);
        }
        return new($"ocr_{Guid.NewGuid():N}", DateTimeOffset.UtcNow.ToUnixTimeSeconds(), string.Join("\n\n", markdown), processed);
    }

    private static ResponseResult OcrResponse(string model, OcrResult result)
    {
        var output = new { type = "message", id = $"msg_{result.Id}", role = "assistant", status = "completed",
            content = new[] { new { type = "output_text", text = result.Text, annotations = System.Array.Empty<object>() } } };
        return new() { Id = result.Id, Model = model, CreatedAt = result.Created, CompletedAt = result.Created, Status = "completed",
            Output = [JsonSerializer.SerializeToElement(output)], AdditionalProperties = new() { ["mistral"] = JsonSerializer.SerializeToElement(new { pages_processed = result.Pages }) } };
    }

    private static IEnumerable<ResponseStreamPart> OcrResponseEvents(ResponseResult result)
    {
        var item = (JsonElement)result.Output.Single();
        var itemId = String(item, "id");
        var part = item.GetProperty("content")[0];
        var text = String(part, "text");
        yield return Event(new { type = "response.created", sequence_number = 0,
            response = new { id = result.Id, @object = "response", created_at = result.CreatedAt, model = result.Model, status = "in_progress", output = System.Array.Empty<object>() } });
        yield return Event(new { type = "response.output_item.added", sequence_number = 1, output_index = 0,
            item = new { type = "message", id = itemId, role = "assistant", status = "in_progress", content = System.Array.Empty<object>() } });
        yield return Event(new { type = "response.content_part.added", sequence_number = 2, item_id = itemId, output_index = 0, content_index = 0,
            part = new { type = "output_text", text = "", annotations = System.Array.Empty<object>() } });
        yield return Event(new { type = "response.output_text.delta", sequence_number = 3, item_id = itemId, output_index = 0, content_index = 0, delta = text });
        yield return Event(new { type = "response.output_text.done", sequence_number = 4, item_id = itemId, output_index = 0, content_index = 0, text });
        yield return Event(new { type = "response.content_part.done", sequence_number = 5, item_id = itemId, output_index = 0, content_index = 0, part });
        yield return Event(new { type = "response.output_item.done", sequence_number = 6, output_index = 0, item });
        yield return Event(new { type = "response.completed", sequence_number = 7, response = result });

        static ResponseStreamPart Event<T>(T value) => JsonSerializer.SerializeToElement(value, ResponseJson.Default).Deserialize<ResponseStreamPart>(ResponseJson.Default)!;
    }

    private sealed record OcrResult(string Id, long Created, string Text, long Pages);

    private sealed record OcrFile(bool Image, string? Url, string? FileId)
    {
        public JsonObject ToDocument()
        {
            if (Url is null && !string.IsNullOrWhiteSpace(FileId)) return new() { ["type"] = "file", ["file_id"] = FileId };
            // Forward signed URLs verbatim to Mistral. Never download them in the gateway.
            if (Url is null || !(Uri.TryCreate(Url, UriKind.Absolute, out var uri) && uri.IsWellFormedOriginalString()
                && uri.Scheme is "https" or "http" && uri.Host.Length > 0 && uri.UserInfo.Length == 0
                || Url.StartsWith("data:", StringComparison.OrdinalIgnoreCase)
                    && Url.IndexOf(',') is var comma and > 0 && Url.AsSpan(0, comma).EndsWith(";base64", StringComparison.OrdinalIgnoreCase)
                    && comma < Url.Length - 1))
                throw new ProviderRequestException("Mistral OCR requires an HTTP(S) URL, a base64 data URL, or a Mistral file ID.");
            var field = Image ? "image_url" : "document_url";
            return new() { ["type"] = field, [field] = Url };
        }
    }
}
