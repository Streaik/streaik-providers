using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal sealed partial class GoogleGeminiProvider
{
    internal const string InteractionReplayKey = "googleInteractionsReplay";

    private sealed class InteractionChatSteps
    {
        private readonly SortedDictionary<int, JsonObject> steps = [];
        private readonly Dictionary<int, StringBuilder> arguments = [];
        private readonly Dictionary<int, StringBuilder> text = [];
        private readonly Dictionary<int, JsonArray> summaries = [];

        public void Observe(JsonElement item, string? type)
        {
            if (Property(item, "index") is not { ValueKind: JsonValueKind.Number } slot || !slot.TryGetInt32(out var index)) return;
            if ((type is "step.start" or "step.stop") && Property(item, "step") is { ValueKind: JsonValueKind.Object } step)
            {
                if (!steps.TryGetValue(index, out var current)) steps[index] = (JsonObject)Copy(step)!;
                else foreach (var field in step.EnumerateObject()) current[field.Name] = Copy(field.Value);
            }
            if (type != "step.delta") return;
            var delta = Property(item, "delta");
            switch (String(delta, "type"))
            {
                case "function_call" or "arguments_delta" when steps.TryGetValue(index, out var call):
                    if (String(delta, "signature") is { Length: > 0 } signature) call["signature"] = signature;
                    var args = Property(delta, "arguments");
                    if (args.ValueKind == JsonValueKind.Object) call["arguments"] = Copy(args);
                    if (args.ValueKind == JsonValueKind.String) Buffer(arguments, index).Append(args.GetString());
                    break;
                case "thought_signature" when String(delta, "signature") is { Length: > 0 } thoughtSignature:
                    Step(index, "thought")["signature"] = thoughtSignature;
                    break;
                case "thought_summary" when Property(delta, "content") is { ValueKind: JsonValueKind.Object } content:
                    Step(index, "thought");
                    if (!summaries.TryGetValue(index, out var summary)) summaries[index] = summary = [];
                    summary.Add(Copy(content));
                    break;
                case "text" when String(delta, "text") is { Length: > 0 } chunk:
                    Step(index, "model_output");
                    Buffer(text, index).Append(chunk);
                    break;
            }
        }

        public JsonElement[] Complete(JsonElement result)
        {
            var terminal = Array(Property(result, "steps")).ToArray();
            var positions = terminal.Select((step, index) => (Key: Key(step), Index: index)).GroupBy(entry => entry.Key)
                .ToDictionary(group => group.Key, group => new Queue<int>(group.Select(entry => entry.Index)));
            var matched = new Dictionary<int, int>();
            var observed = steps.ToArray();
            var projected = new JsonElement[observed.Length];
            for (var i = 0; i < observed.Length; i++)
            {
                var (index, step) = observed[i];
                var type = (string?)step["type"];
                var canonical = default(JsonElement);
                var key = (type, type == "function_call" ? (string?)step["id"] : null);
                if (positions.TryGetValue(key, out var candidates) && candidates.TryDequeue(out var position))
                {
                    canonical = terminal[position];
                    matched.Add(position, i);
                    foreach (var field in canonical.EnumerateObject())
                        if (field.Value.ValueKind != JsonValueKind.Null) step[field.Name] = Copy(field.Value);
                }
                // Deltas are the full arguments, not a suffix of the JSON at step.start.
                // Use a non-empty final object when present, otherwise rebuild from the deltas.
                var finalArgs = Property(canonical, "arguments");
                if (arguments.TryGetValue(index, out var args) && args.Length > 0
                    && (finalArgs.ValueKind != JsonValueKind.Object || !finalArgs.EnumerateObject().Any()))
                    step["arguments"] = ParseArguments(args.ToString());
                if (summaries.TryGetValue(index, out var summary) && step["summary"] is not JsonArray { Count: > 0 }) step["summary"] = summary;
                if (text.TryGetValue(index, out var content) && step["content"] is not JsonArray { Count: > 0 })
                    step["content"] = new JsonArray(TextPart(content.ToString()));
                projected[i] = JsonSerializer.SerializeToElement(step, Json);
            }

            if (matched.Count == observed.Length)
                return terminal.Select((step, index) => matched.TryGetValue(index, out var i) ? projected[i] : step).ToArray();
            // A sparse completion must not discard already assembled steps. Insert any terminal-only
            // steps before their next known neighbor, leaving other streamed steps in index order.
            var merged = projected.ToList();
            for (var i = 0; i < terminal.Length; i++)
            {
                if (matched.ContainsKey(i)) continue;
                var next = i + 1;
                while (next < terminal.Length && !matched.ContainsKey(next)) next++;
                if (next < terminal.Length) merged.Insert(merged.IndexOf(projected[matched[next]]), terminal[i]);
                else merged.Add(terminal[i]);
            }
            return merged.ToArray();

            static (string? Type, string? CallId) Key(JsonElement step) => String(step, "type") is { } type
                ? (type, type == "function_call" ? String(step, "id") : null)
                : (null, null);
        }

        private JsonObject Step(int index, string type)
            => steps.TryGetValue(index, out var step) ? step : steps[index] = new JsonObject { ["type"] = type };

        private static StringBuilder Buffer(Dictionary<int, StringBuilder> buffers, int index)
            => buffers.TryGetValue(index, out var buffer) ? buffer : buffers[index] = new();
    }

    private static Dictionary<string, JsonElement>? CaptureInteractionReplay(string model, JsonElement[] steps)
    {
        var replay = steps.Where(step => String(step, "type") is "thought" or "model_output" or "function_call").ToArray();
        if (!replay.Any(step => String(step, "type") == "thought")
            && !replay.Skip(1).Any(step => String(step, "type") == "model_output"))
            return null;
        return new() { [InteractionReplayKey] = JsonSerializer.SerializeToElement(new { model, steps = replay }, Json) };
    }

    // A returned snapshot supplies native thought steps and boundaries, never extra client calls or results.
    // The original UI message and its checked approvals still own the text and function call arguments.
    private static List<JsonNode>? TryReplayInteractionRound(JsonElement snapshot, string model, JsonArray output, List<JsonObject> calls)
    {
        if (String(snapshot, "model") != model || Property(snapshot, "steps") is not { ValueKind: JsonValueKind.Array } steps
            || output.Any(part => (string?)part?["type"] != "text")) return null;
        var replay = new List<JsonNode>();
        var text = new StringBuilder();
        var callIndex = 0;
        foreach (var step in Array(steps))
        {
            switch (String(step, "type"))
            {
                case "thought":
                    replay.Add(Copy(step)!);
                    break;
                case "model_output" when Property(step, "content") is { ValueKind: JsonValueKind.Array } content:
                    foreach (var part in Array(content))
                    {
                        if (String(part, "type") != "text" || String(part, "text") is not { } value) return null;
                        text.Append(value);
                    }
                    replay.Add(Copy(step)!);
                    break;
                case "function_call" when callIndex < calls.Count:
                    var call = calls[callIndex++];
                    if (String(step, "id") != (string?)call["id"] || String(step, "name") != (string?)call["name"]
                        || String(step, "signature") != (string?)call["signature"]
                        || !JsonNode.DeepEquals(call["arguments"], Property(step, "arguments") is { ValueKind: JsonValueKind.Object } args ? Copy(args) : new JsonObject())) return null;
                    replay.Add(call);
                    break;
                default: return null;
            }
        }
        return callIndex == calls.Count && text.ToString() == string.Concat(output.Select(part => (string?)part?["text"])) ? replay : null;
    }
}
