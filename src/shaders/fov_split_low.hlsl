// Split foveation, after the model: what the tone transfer compares, at the low resolution, as
// the model's change -- log(answer) - log(input) per channel -- rather than raw colour, so a small
// bright feature cannot dominate a neighbourhood's average:
//   global: the low-res frame's change (its answer against the input it was shown)
//   crop:   the crop's change averaged over the same footprint (edge-repeated outside the crop;
//           only its inside is ever used)
#include "fov_common.hlsli"

Texture2D<float4> Answer : register(t0);   // the model's atlas
Texture2D<float4> Proxy : register(t1);    // the model's input, full size
RWTexture2D<float4> LowGlobal : register(u0);
RWTexture2D<float4> LowInput : register(u1);
RWTexture2D<float4> LowCrop : register(u2);

static const float kFloor = 0.01;   // keeps near-black channels from producing wild logs

float3 Change(float3 answer, float3 input)
{
    return log(max(answer, 0.0) + kFloor) - log(max(input, 0.0) + kFloor);
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_low_w || id.y >= g_low_h)
        return;

    float2 scale = float2(g_full_w, g_full_h) / float2(g_low_w, g_low_h);
    float2 full = ((float2)id.xy + 0.5) * scale;
    float4 input = BoxSample(Proxy, full, scale);
    LowInput[id.xy] = input;
    LowGlobal[id.xy] = float4(Change(Answer[uint2(g_low_x + id.x, id.y)].rgb, input.rgb), 1.0);

    // The crop's change at four taps over the same footprint, kept on the crop's texel centres.
    float2 inv_atlas = 1.0 / float2(g_work_w, g_work_h);
    float2 inv_full = 1.0 / float2(g_full_w, g_full_h);
    float2 lo = 0.5, hi = float2(g_crop_w, g_crop_h) - 0.5;
    float2 o = scale * 0.25;
    float2 origin = float2(g_crop_x, g_crop_y);
    float3 sum = 0;
    [unroll] for (int i = 0; i < 4; ++i)
    {
        float2 c = clamp(full + float2((i & 1) ? o.x : -o.x, (i & 2) ? o.y : -o.y) - origin, lo, hi);
        sum += Change(Answer.SampleLevel(Linear, c * inv_atlas, 0).rgb,
                      Proxy.SampleLevel(Linear, (c + origin) * inv_full, 0).rgb);
    }
    LowCrop[id.xy] = float4(0.25 * sum, 1.0);
}
