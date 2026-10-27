// Split foveation: a light separable Gaussian over the low-resolution offsets (it only softens the
// low-res grid). The vertical pass writes the two offsets the expand adds: the whole-frame pass's
// (for the periphery) and that minus the crop's (for the fovea), so both regions take their coarse
// brightness and colour from the same pass.
#include "fov_common.hlsli"

Texture2D<float4> InGlobal : register(t0);
Texture2D<float4> InInput : register(t1);
Texture2D<float4> InCrop : register(t2);
RWTexture2D<float4> OutA : register(u0);
RWTexture2D<float4> OutB : register(u1);
RWTexture2D<float4> OutC : register(u2);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_low_w || id.y >= g_low_h)
        return;

    int2 step = g_blur_axis == 0 ? int2(1, 0) : int2(0, 1);
    int2 last = int2(g_low_w - 1, g_low_h - 1);
    int r = (int)ceil(g_sigma * 3.0);
    float k2 = -0.5 / (g_sigma * g_sigma);
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
        OutA[id.xy] = float4(gl.rgb, 0.0);
        OutB[id.xy] = float4(gl.rgb - cr.rgb, 0.0);
    }
}
