// Split foveation, before the model: one atlas holding a full-resolution crop around the gaze
// (left) and the whole frame at low resolution (from g_low_x). Pixels between and below the two
// regions repeat the nearest edge, so the model sees no hard border there. Motion vectors are in
// atlas pixels; the crop's are measured against last frame's crop, so its history follows it.
#include "fov_common.hlsli"

Texture2D<float4> Proxy : register(t0);
Texture2D<float> Depth : register(t1);
Texture2D<float2> Motion : register(t2);
RWTexture2D<float4> ProxyOut : register(u0);
RWTexture2D<float> DepthOut : register(u1);
RWTexture2D<float2> MotionOut : register(u2);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_work_w || id.y >= g_work_h)
        return;

    const bool crop = id.x < g_crop_w + (g_low_x - g_crop_w) / 2;
    float2 full;
    float2 scale;   // full pixels per atlas pixel
    if (crop)
    {
        uint2 u = min(id.xy, uint2(g_crop_w - 1, g_crop_h - 1));
        full = float2(g_crop_x, g_crop_y) + (float2)u + 0.5;
        scale = 1.0;
        ProxyOut[id.xy] = Proxy[uint2(full)];
    }
    else
    {
        uint2 v = min(uint2((uint)max((int)id.x - (int)g_low_x, 0), id.y), uint2(g_low_w - 1, g_low_h - 1));
        scale = float2(g_full_w, g_full_h) / float2(g_low_w, g_low_h);
        full = ((float2)v + 0.5) * scale;
        ProxyOut[id.xy] = BoxSample(Proxy, full, scale);
    }

    float2 to_render = float2(g_render_w, g_render_h) / float2(g_full_w, g_full_h);
    uint2 rp = min((uint2)(full * to_render), uint2(g_render_w - 1, g_render_h - 1));
    DepthOut[id.xy] = Depth[rp];

    // Game convention: previous position = current + mv * scale, in render pixels.
    float2 mv_full = Motion[rp] * float2(g_mv_scale_x, g_mv_scale_y) / to_render;
    if (crop)
        MotionOut[id.xy] = mv_full + float2((float)g_crop_x - (float)g_prev_crop_x,
                                            (float)g_crop_y - (float)g_prev_crop_y);
    else
        MotionOut[id.xy] = mv_full / scale;
}
