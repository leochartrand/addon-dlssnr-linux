// Split foveation, after the model: the three pictures the tone transfer compares, all at the
// low resolution -- the model's answer for the whole frame, the model's input, and the crop's
// answer (repeating its edge outside the crop; only its inside is ever used).
#include "fov_common.hlsli"

Texture2D<float4> Answer : register(t0);   // the model's atlas
Texture2D<float4> Proxy : register(t1);    // the model's input, full size
RWTexture2D<float4> LowGlobal : register(u0);
RWTexture2D<float4> LowInput : register(u1);
RWTexture2D<float4> LowCrop : register(u2);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_low_w || id.y >= g_low_h)
        return;

    LowGlobal[id.xy] = Answer[uint2(g_low_x + id.x, id.y)];

    float2 scale = float2(g_full_w, g_full_h) / float2(g_low_w, g_low_h);
    float2 full = ((float2)id.xy + 0.5) * scale;
    LowInput[id.xy] = BoxSample(Proxy, full, scale);

    // The same footprint over the crop's answer, kept on the crop's own texel centres.
    float2 inv = 1.0 / float2(g_work_w, g_work_h);
    float2 lo = 0.5, hi = float2(g_crop_w, g_crop_h) - 0.5;
    float2 o = scale * 0.25;
    float2 c = full - float2(g_crop_x, g_crop_y);
    LowCrop[id.xy] = 0.25 * (Answer.SampleLevel(Linear, clamp(c + float2(-o.x, -o.y), lo, hi) * inv, 0) +
                             Answer.SampleLevel(Linear, clamp(c + float2(o.x, -o.y), lo, hi) * inv, 0) +
                             Answer.SampleLevel(Linear, clamp(c + float2(-o.x, o.y), lo, hi) * inv, 0) +
                             Answer.SampleLevel(Linear, clamp(c + float2(o.x, o.y), lo, hi) * inv, 0));
}
