// Debug view 4, drawn over the finished frame: the crop (green, full resolution) and the rest (red, which
// only the low-res pass saw), a cyan cross at the fovea centre and a ring at the raw gaze point.
#include "fov_common.hlsli"

Texture2D<float4> Unused0 : register(t0);
RWTexture2D<float4> Frame : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_full_w || id.y >= g_full_h)
        return;

    int2 q = (int2)id.xy - int2(CropOrigin());
    bool in_crop = all(q >= 0) && all(q < int2(g_crop_w, g_crop_h));
    float3 tint = in_crop ? float3(0.1, 0.9, 0.2) : float3(0.95, 0.15, 0.1);
    float4 c = Frame[id.xy];
    c.rgb = lerp(c.rgb, tint, 0.35);

    float2 p = (float2)id.xy + 0.5;
    float2 dc = abs(p - float2(g_centre_x, g_centre_y));
    if ((dc.x < 1.5 && dc.y < 24.0) || (dc.y < 1.5 && dc.x < 24.0))
        c.rgb = float3(0.0, 0.9, 1.0);
    if (g_gaze_x >= 0.0)
    {
        float r = length(p - float2(g_gaze_x, g_gaze_y));
        if (r < 14.0) c.rgb = r < 10.0 ? float3(1.0, 1.0, 1.0) : float3(0.0, 0.0, 0.0);
    }
    Frame[id.xy] = c;
}
