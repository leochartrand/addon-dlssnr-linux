// Split foveation: the full-size answer. Outside the crop it is the model's input carrying the
// whole-frame pass's broad change; inside, the crop's answer corrected by the difference between
// the two passes' broad changes, blended in over g_feather pixels from the crop's edge. The log
// gains are capped at 2x either way.
#include "fov_common.hlsli"

Texture2D<float4> Answer : register(t0);
Texture2D<float4> Proxy : register(t1);
Texture2D<float4> GainPeriphery : register(t2);   // log gains
Texture2D<float4> GainFovea : register(t4);
RWTexture2D<float4> Out : register(u0);

static const float kMaxLogGain = 0.693;

float3 Gain(Texture2D<float4> t, float2 uv)
{
    return exp(clamp(t.SampleLevel(Linear, uv, 0).rgb, -kMaxLogGain, kMaxLogGain));
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_full_w || id.y >= g_full_h)
        return;

    float2 uv = ((float2)id.xy + 0.5) / float2(g_full_w, g_full_h);
    float4 base = Proxy[id.xy];
    float4 result = float4(base.rgb * Gain(GainPeriphery, uv), base.a);

    int2 c = (int2)id.xy - int2(g_crop_x, g_crop_y);
    int2 size = int2(g_crop_w, g_crop_h);
    if (all(c >= 0) && all(c < size))
    {
        float2 d2 = min((float2)c + 0.5, (float2)(size - c) - 0.5);
        float d = min(d2.x, d2.y);
        float w = g_feather > 0.0 ? smoothstep(0.0, g_feather, d) : 1.0;
        float4 a = Answer[uint2(c)];
        float4 fovea = float4(a.rgb * Gain(GainFovea, uv), a.a);
        result = lerp(result, fovea, w);
    }
    Out[id.xy] = result;
}
