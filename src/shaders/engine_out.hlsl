// The NR engine's answer for one region of the atlas: its head, float32 [4, field_h, field_w] (the composed
// frame and the temporal-blend logit), blended with the reprojected history by the network's own weight;
// the answer is also the region's next history, truncated toward zero to the half grid.
#include "engine_common.hlsli"

RWByteAddressBuffer Head : register(u0);
RWTexture2D<float4> Answer : register(u1);
RWTexture2D<float4> HistNext : register(u2);

float TruncHalf(float x)
{
    uint h = f32tof16(x);   // round to nearest; step back one ulp where that went up in magnitude
    if (abs(f16tof32(h)) > abs(x))
        h -= 1;
    return f16tof32(h);
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_w || id.y >= g_h)
        return;
    uint plane = g_field_w * g_field_h;
    uint i = id.y * g_field_w + id.x;
    float3 neural = float3(asfloat(Head.Load(i * 4)), asfloat(Head.Load((plane + i) * 4)),
                           asfloat(Head.Load((2 * plane + i) * 4)));
    float logit = asfloat(Head.Load((3 * plane + i) * 4));
    uint2 p = uint2(g_x + id.x, g_y + id.y);
    float2 prev;
    if (Reproject(p, prev))
    {
        float weight = saturate(g_blend / (1.0 + exp(-logit)));
        neural = lerp(neural, History(prev), weight);
    }
    neural = saturate(neural);
    float3 t = float3(TruncHalf(neural.r), TruncHalf(neural.g), TruncHalf(neural.b));
    Answer[p] = float4(t, 1.0);
    HistNext[p] = float4(t, 1.0);
}
