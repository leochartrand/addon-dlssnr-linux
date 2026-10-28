// Split foveation: a light separable Gaussian over the low-resolution offsets (it only softens the
// low-res grid). The vertical pass writes the two offsets the expand adds: the whole-frame pass's
// (for the periphery) and that minus the crop's (for the fovea), so both regions take their coarse
// brightness and colour from the same pass. The whole-frame offset is also blended over time with
// last frame's, carried along the motion vectors, and gated by the game's own picture: where the
// input (blurred, reprojected) has not changed beyond a dead band, the tone barely moves, so the
// model's frame-to-frame noise -- it re-draws its detail from fresh noise every frame -- and small
// animation do not flicker it; where the input really changes, the tone follows at once. On frames
// where the whole-frame pass did not run, the carried history is used alone. The history keeps the
// tone in rgb and the blurred input's luma in a. The offsets the expand adds carry that tone limited by
// the relight settings.
#include "fov_common.hlsli"

Texture2D<float4> InGlobal : register(t0);
Texture2D<float4> InInput : register(t1);
Texture2D<float4> InCrop : register(t2);
Texture2D<float4> History : register(t4);   // V pass: last frame's whole-frame offset
Texture2D<float2> Motion : register(t5);    // V pass: the game's motion vectors, render size
RWTexture2D<float4> OutA : register(u0);
RWTexture2D<float4> OutB : register(u1);
RWTexture2D<float4> OutC : register(u2);

float Brightness(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_low_w || id.y >= g_low_h)
        return;

    int2 step = g_blur_axis == 0 ? int2(1, 0) : int2(0, 1);
    int2 last = int2(g_low_w - 1, g_low_h - 1);
    int r = (int)ceil(g_sigma * 3.0);
    float k2 = g_sigma > 0.0 ? -0.5 / (g_sigma * g_sigma) : 0.0;   // 0: no blur (r = 0)
    float4 gl = 0, in_ = 0, cr = 0;
    float wsum = 0;
    for (int k = -r; k <= r; ++k)
    {
        float w = exp(k * k * k2);
        int2 q = clamp((int2)id.xy + step * k, int2(0, 0), last);
        gl += w * InGlobal[q];
        in_ += w * InInput[q];
        cr += w * InCrop[q];
        wsum += w;
    }
    gl /= wsum;
    in_ /= wsum;
    cr /= wsum;

    if (g_blur_axis == 0)
    {
        OutA[id.xy] = gl;
        OutB[id.xy] = in_;
        OutC[id.xy] = cr;
    }
    else
    {
        float3 tone = gl.rgb;
        if (g_tone_alpha < 1.0)
        {
            // Game convention: previous position = current + mv * scale, in render pixels.
            float2 full = ((float2)id.xy + 0.5) * float2(g_full_w, g_full_h) / float2(g_low_w, g_low_h);
            float2 to_render = float2(g_render_w, g_render_h) / float2(g_full_w, g_full_h);
            uint2 rp = min((uint2)(full * to_render), uint2(g_render_w - 1, g_render_h - 1));
            float2 prev_uv = (full + Motion[rp] * float2(g_mv_scale_x, g_mv_scale_y) / to_render) /
                             float2(g_full_w, g_full_h);
            bool on_screen = all(prev_uv >= 0.0) && all(prev_uv <= 1.0);
            float4 hist = History.SampleLevel(Linear, prev_uv, 0);
            float change = abs(Brightness(in_.rgb) - hist.a);
            float follow = smoothstep(g_tone_deadband, 4.0 * g_tone_deadband, change);
            float a = g_tone_alpha == 0.0 ? 0.0 : lerp(g_tone_alpha, 1.0, follow);
            if (!on_screen && g_tone_alpha > 0.0)
                a = 1.0;
            tone = lerp(hist.rgb, gl.rgb, a);
        }
        // The history keeps the model's own tone; the picture gets it limited (g_relight, g_relight_mode).
        // Relight 0: the whole-frame pass did not run (details only); its region of the answer is stale.
        float3 shown = g_relight > 0.0 ? tone * g_relight : 0.0;
        if (g_relight_mode == 1)
            shown = min(Brightness(shown), 0.0).xxx;
        OutA[id.xy] = float4(shown, 0.0);
        OutB[id.xy] = float4(shown - cr.rgb, 0.0);
        OutC[id.xy] = float4(tone, Brightness(in_.rgb));
    }
}
