// Split foveation: the full-size answer. Outside the crop: the input plus the whole-frame pass's
// offset -- the low-res answer upsampled, with the input's own finer detail on top. Inside: the
// crop's answer with its coarse part swapped for the low-res answer's, blended in over g_feather
// pixels from the crop's edge.
// g_pass_debug shows one pass alone, as the model saw and answered it: 1-3 the fovea crop (what it
// sees, its answer, the change x20; outside the crop the input, dimmed), 4-6 the same for the
// low-res whole-frame pass, stretched over the screen.
#include "fov_common.hlsli"

Texture2D<float4> Answer : register(t0);
Texture2D<float4> Proxy : register(t1);
Texture2D<float4> OffsetPeriphery : register(t2);
Texture2D<float4> OffsetFovea : register(t4);
Texture2D<float4> InputAtlas : register(t5);   // what the model was given: crop and low-res frame
RWTexture2D<float4> Out : register(u0);

float4 PassView(uint view, float4 seen, float4 answer)
{
    if (view == 1) return seen;
    if (view == 2) return answer;
    return float4(saturate(abs(answer.rgb - seen.rgb) * 20.0), 1.0);
}

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
    if (g_pass_debug >= 4)
    {
        float2 a = (float2(g_low_x, 0.0) + uv * float2(g_low_w, g_low_h)) / float2(g_work_w, g_work_h);
        Out[id.xy] = PassView(g_pass_debug - 3, InputAtlas.SampleLevel(Linear, a, 0),
                              Answer.SampleLevel(Linear, a, 0));
        return;
    }
    if (g_pass_debug >= 1)
    {
        int2 q = (int2)id.xy - int2(g_crop_x, g_crop_y);
        bool inside = all(q >= 0) && all(q < int2(g_crop_w, g_crop_h));
        Out[id.xy] = inside ? PassView(g_pass_debug, InputAtlas[uint2(q)], Answer[uint2(q)])
                            : float4(base.rgb * 0.3, base.a);
        return;
    }
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
