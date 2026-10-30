// Split foveation, the crop on top of the periphery: before the crop's model call, its input becomes the frame
// plus the periphery pass's change (the low-res answer minus what it saw, stretched edge-aware as the expand
// does), so the crop refines the periphery's picture instead of grading the raw frame in parallel with it.
#include "fov_common.hlsli"

RWTexture2D<float4> Atlas : register(u0);    // the model's input: the crop (rewritten here) and the low-res frame
RWTexture2D<float4> Answer : register(u1);   // the periphery's answer, at g_low_x (read only)

float Brightness(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_crop_w || id.y >= g_crop_h)
        return;
    const float4 base = Atlas[id.xy];
    const float2 uv = (CropOrigin() + (float2)id.xy + 0.5) / float2(g_full_w, g_full_h);
    // The 4x4 low-res texels around it, by distance and by brightness likeness (fov_split_expand's Offsets).
    const float2 pos = uv * float2(g_low_w, g_low_h) - 0.5;
    const int2 i0 = (int2)floor(pos);
    const int2 last = int2(g_low_w - 1, g_low_h - 1);
    const float y = Brightness(base.rgb);
    float3 sum = 0;
    float wsum = 0;
    [unroll] for (int dy = -1; dy <= 2; ++dy)
    {
        [unroll] for (int dx = -1; dx <= 2; ++dx)
        {
            const int2 q = clamp(i0 + int2(dx, dy), int2(0, 0), last);
            const float2 d = pos - (float2)(i0 + int2(dx, dy));
            const uint2 at = uint2(g_low_x + q.x, q.y);
            const float3 seen = Atlas[at].rgb;
            const float g = Brightness(seen) - y;
            const float w = exp(-dot(d, d) * 0.889) * (exp(-g * g * 312.5) + 1e-3);
            sum += w * (Answer[at].rgb - seen);
            wsum += w;
        }
    }
    Atlas[id.xy] = float4(saturate(base.rgb + sum / wsum), base.a);
}
