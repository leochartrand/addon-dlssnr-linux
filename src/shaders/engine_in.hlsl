// The NR engine's inputs for one region of the atlas: the frame, float32 [3, field_h, field_w], and the
// history (features 7-9): last frame's answer reprojected, or the frame itself where there is none. The
// region's last row and column are mirrored out to the field (dlssnr frame_to_tensor). The proxy goes in
// as the half floats the model is given, as NVIDIA's library reads it.
#include "engine_common.hlsli"

RWByteAddressBuffer Image : register(u0);
RWByteAddressBuffer Hist : register(u3);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_field_w || id.y >= g_field_h)
        return;
    uint2 p = uint2(g_x + Mirror(id.x, g_w), g_y + Mirror(id.y, g_h));
    float3 c = saturate(Proxy[p].rgb);
    float3 h = c;
    float2 prev;
    if (Reproject(p, prev))
        h = History(prev);
    uint plane = g_field_w * g_field_h;
    uint i = id.y * g_field_w + id.x;
    [unroll] for (uint k = 0; k < 3; ++k)
    {
        Image.Store((k * plane + i) * 4, asuint(c[k]));
        Hist.Store((k * plane + i) * 4, asuint(h[k]));
    }
}
