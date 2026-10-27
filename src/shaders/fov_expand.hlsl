// Stretches the model's working-resolution answer back over the full frame, for the usual resolve.
#include "fov_common.hlsli"

Texture2D<float4> Answer : register(t0);
RWTexture2D<float4> Out : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_full_w || id.y >= g_full_h)
        return;
    float2 work = Forward(g_fwd, (float2)id.xy + 0.5);
    Out[id.xy] = Answer.SampleLevel(Linear, work / float2(g_work_w, g_work_h), 0);
}
