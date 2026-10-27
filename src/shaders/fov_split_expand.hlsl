// Split foveation: the full-size answer. Outside the crop: the input plus the whole-frame pass's
// offset -- the low-res answer upsampled, with the input's own finer detail on top. Inside: the
// crop's answer with its coarse part swapped for the low-res answer's, blended in over g_feather
// pixels from the crop's edge.
#include "fov_common.hlsli"

Texture2D<float4> Answer : register(t0);
Texture2D<float4> Proxy : register(t1);
Texture2D<float4> OffsetPeriphery : register(t2);
Texture2D<float4> OffsetFovea : register(t4);
RWTexture2D<float4> Out : register(u0);

float3 Offset(Texture2D<float4> t, float2 uv)
{
    return t.SampleLevel(Linear, uv, 0).rgb;
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_full_w || id.y >= g_full_h)
        return;

    float2 uv = ((float2)id.xy + 0.5) / float2(g_full_w, g_full_h);
    float4 base = Proxy[id.xy];
    float4 result = float4(saturate(base.rgb + Offset(OffsetPeriphery, uv)), base.a);

    int2 c = (int2)id.xy - int2(g_crop_x, g_crop_y);
    int2 size = int2(g_crop_w, g_crop_h);
    if (all(c >= 0) && all(c < size))
    {
        float2 d2 = min((float2)c + 0.5, (float2)(size - c) - 0.5);
        float d = min(d2.x, d2.y);
        float w = g_feather > 0.0 ? smoothstep(0.0, g_feather, d) : 1.0;
        float4 a = Answer[uint2(c)];
        float4 fovea = float4(saturate(a.rgb + Offset(OffsetFovea, uv)), a.a);
        result = lerp(result, fovea, w);
    }
    Out[id.xy] = result;
}
