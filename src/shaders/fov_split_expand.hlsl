// Split foveation: the full-size answer. Outside the crop: the input plus the whole-frame pass's
// offset -- the low-res answer upsampled edge-aware, with the input's own finer detail on top. Inside: the
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
Texture2D<float> Depth : register(t6);         // the game's depth, render size
RWTexture2D<float4> Out : register(u0);

float4 PassView(uint view, float4 seen, float4 answer)
{
    if (view == 1) return seen;
    if (view == 2) return answer;
    return float4(saturate(abs(answer.rgb - seen.rgb) * 20.0), 1.0);
}

float Brightness(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// The low-resolution offsets at a full-resolution pixel, stretched edge-aware: the 4x4 low-res texels
// around it, each weighted by distance and by how close its input brightness (the low-res frame the
// model saw) is to this pixel's own, so a change made on one side of an edge (a building against the
// sky, a character against a wall) stays on that side instead of glowing across it.
void Offsets(float2 uv, float3 base, bool fovea, out float3 periphery, out float3 crop)
{
    float2 pos = uv * float2(g_low_w, g_low_h) - 0.5;
    int2 i0 = (int2)floor(pos);
    int2 last = int2(g_low_w - 1, g_low_h - 1);
    float y = Brightness(base);
    float3 sp = 0, sc = 0;
    float wsum = 0;
    [unroll] for (int dy = -1; dy <= 2; ++dy)
    {
        [unroll] for (int dx = -1; dx <= 2; ++dx)
        {
            int2 q = clamp(i0 + int2(dx, dy), int2(0, 0), last);
            float2 d = pos - (float2)(i0 + int2(dx, dy));
            float g = Brightness(InputAtlas[uint2(g_low_x + q.x, q.y)].rgb) - y;
            // spatial: Gaussian, sigma 0.75 texels; range: sigma 0.04 in display brightness. The floor
            // keeps a pixel finer than the low-res grid (a flagpole) on its neighbourhood's offset.
            float w = exp(-dot(d, d) * 0.889) * (exp(-g * g * 312.5) + 1e-3);
            sp += w * OffsetPeriphery[q].rgb;
            if (fovea)
                sc += w * OffsetFovea[q].rgb;
            wsum += w;
        }
    }
    periphery = sp / wsum;
    crop = sc / wsum;
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
        int2 q = (int2)id.xy - int2(CropOrigin());
        bool inside = all(q >= 0) && all(q < int2(g_crop_w, g_crop_h));
        Out[id.xy] = inside ? PassView(g_pass_debug, InputAtlas[uint2(q)], Answer[uint2(q)])
                            : float4(base.rgb * 0.3, base.a);
        return;
    }
    int2 c = (int2)id.xy - int2(CropOrigin());
    int2 size = int2(g_crop_w, g_crop_h);
    bool in_crop = all(c >= 0) && all(c < size);
    float3 off_periphery, off_fovea;
    Offsets(uv, base.rgb, in_crop, off_periphery, off_fovea);
    // The sky (depth at the far plane) is never relit: the broad change there is the model inventing
    // weather (dark blobs, halos hugging the skyline), and it is nothing ambient occlusion could do.
    float2 to_render = float2(g_render_w, g_render_h) / float2(g_full_w, g_full_h);
    uint2 rp = min((uint2)(((float2)id.xy + 0.5) * to_render), uint2(g_render_w - 1, g_render_h - 1));
    float z = Depth[rp];
    float sky = g_depth_inverted != 0 ? 1.0 - smoothstep(0.0, 1e-5, z) : smoothstep(1.0 - 1e-5, 1.0, z);
    off_fovea -= sky * off_periphery;    // the crop keeps its fine detail, without the broad change
    off_periphery *= 1.0 - sky;
    float4 result = float4(saturate(base.rgb + off_periphery), base.a);

    if (in_crop)
    {
        float2 d2 = min((float2)c + 0.5, (float2)(size - c) - 0.5);
        float d = min(d2.x, d2.y);
        float w = g_feather > 0.0 ? smoothstep(0.0, g_feather, d) : 1.0;
        float4 a = Answer[uint2(c)];
        float4 fovea = float4(saturate(a.rgb + off_fovea), a.a);
        result = lerp(result, fovea, w);
    }
    Out[id.xy] = result;
}
