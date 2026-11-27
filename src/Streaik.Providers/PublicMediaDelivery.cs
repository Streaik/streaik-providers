using System.Net;
using System.Net.Sockets;
using Microsoft.Extensions.DependencyInjection;
using static Streaik.Providers.MediaHttp;

namespace Streaik.Providers;

internal static class PublicMediaDelivery
{
    internal static void AddClient(IServiceCollection services, string name)
        => services.AddHttpClient(name).ConfigurePrimaryHttpMessageHandler(() => CreateHandler());

    internal static async Task<(string Data, string Mime)> DownloadAsync(HttpClient client, string url, string fallback, int limit, CancellationToken ct)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, RequirePublicHttpsUri(url));
        using var result = await NativeHttp.SendAsync(client, request, ct);
        var mime = result.Content.Headers.ContentType?.MediaType;
        if (mime is null or "application/octet-stream") mime = fallback;
        if (!mime.StartsWith(fallback[..(fallback.IndexOf('/') + 1)], StringComparison.OrdinalIgnoreCase))
            throw new IOException("Unexpected provider asset type.");
        if (result.Content.Headers.ContentLength > limit) throw new ProviderRequestException("Media exceeds the gateway's inline size limit.", 413);
        await using var stream = await result.Content.ReadAsStreamAsync(ct);
        return (await BoundedBase64Async(stream, limit, ct), mime.ToLowerInvariant());
    }

    internal static Uri RequirePublicHttpsUri(string? url)
        => Uri.TryCreate(url, UriKind.Absolute, out var uri) && uri.Scheme == Uri.UriSchemeHttps && uri.IsDefaultPort
            && uri.HostNameType == UriHostNameType.Dns && uri.UserInfo.Length == 0 && uri.Fragment.Length == 0
                ? uri : throw new IOException("Untrusted provider media URL.");

    internal static SocketsHttpHandler CreateHandler() => new()
    {
        UseCookies = false, AllowAutoRedirect = false, UseProxy = false, ConnectTimeout = TimeSpan.FromSeconds(15),
        // Connect to the addresses we validated, so DNS cannot change the destination between validation and connect.
        ConnectCallback = async (context, ct) =>
        {
            foreach (var address in (await Dns.GetHostAddressesAsync(context.DnsEndPoint.Host, ct)).Where(IsPublicAddress))
            {
                var socket = new Socket(address.AddressFamily, SocketType.Stream, ProtocolType.Tcp);
                try
                {
                    await socket.ConnectAsync(new IPEndPoint(address, context.DnsEndPoint.Port), ct);
                    return new NetworkStream(socket, ownsSocket: true);
                }
                catch (SocketException) { socket.Dispose(); }
                catch { socket.Dispose(); throw; }
            }
            throw new IOException("Could not connect to a public provider media address.");
        }
    };

    private static bool IsPublicAddress(IPAddress address)
    {
        if (address.IsIPv4MappedToIPv6) address = address.MapToIPv4();
        var b = address.GetAddressBytes();
        return address.AddressFamily == AddressFamily.InterNetwork
            ? b[0] is not (0 or 10 or 127 or >= 224)
                && !(b[0] == 100 && b[1] is >= 64 and <= 127) && !(b[0] == 169 && b[1] == 254)
                && !(b[0] == 172 && b[1] is >= 16 and <= 31) && !(b[0] == 192 && b[1] is 0 or 168)
                && !(b[0] == 198 && b[1] is 18 or 19 or 51) && !(b[0] == 203 && b[1] == 0 && b[2] == 113)
            : (b[0] & 0xE0) == 0x20 && !(b[0] == 0x20 && (b[1] == 2 || b[1] == 1 && (b[2] < 2 || b[2] == 0x0D && b[3] == 0xB8)))
                && !(b[0] == 0x3F && b[1] == 0xFF);
    }
}
