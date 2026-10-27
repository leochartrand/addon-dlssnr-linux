// Builds the model's working-resolution inputs from the full-resolution proxy and the game's
// guides. Motion vectors are re-expressed in working pixels through BOTH mappings: the previous
// position of a pixel is found in full resolution, then placed with last frame's mapping, so the
// model's history stays aligned while the fovea moves.
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

    float2 full = Inverse(id.xy);
    ProxyOut[id.xy] = Proxy.SampleLevel(Linear, full / float2(g_full_w, g_full_h), 0);

    float2 to_render = float2(g_render_w, g_render_h) / float2(g_full_w, g_full_h);
    uint2 rp = min((uint2)(full * to_render), uint2(g_render_w - 1, g_render_h - 1));
    DepthOut[id.xy] = Depth[rp];

    // Game convention: previous position = current + mv * scale, in render pixels.
    float2 mv_full = Motion[rp] * float2(g_mv_scale_x, g_mv_scale_y) / to_render;
    float2 prev_work = Forward(g_prev, full + mv_full);
    MotionOut[id.xy] = prev_work - ((float2)id.xy + 0.5);
}
