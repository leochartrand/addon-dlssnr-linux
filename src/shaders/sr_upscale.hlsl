// Foveated super resolution, periphery: the game's render-resolution frame stretched to the output with a
// bilinear filter, its sub-pixel jitter taken out (a plain upscale of a jittered frame would wobble). The crop
// around the gaze is DLSS's, written over this afterwards.
Texture2D<float4> Src : register(t0);
Texture2D<float4> Unused : register(t1);
RWTexture2D<float4> Dst : register(u0);

cbuffer Constants : register(b0)
{
    uint g_src_w;
    uint g_src_h;
    uint g_dst_w;
    uint g_dst_h;
    float g_jitter_x;   // the frame's jitter in render pixels, times the sign that removes it
    float g_jitter_y;
    uint g_pad0;
    uint g_pad1;
};

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_dst_w || id.y >= g_dst_h)
        return;
    const float2 scale = float2(g_src_w, g_src_h) / float2(g_dst_w, g_dst_h);
    const float2 p = (float2(id.xy) + 0.5) * scale - 0.5 + float2(g_jitter_x, g_jitter_y);
    const float2 f = frac(p);
    const int2 i = int2(floor(p));
    const int2 hi = int2(g_src_w, g_src_h) - 1;
    const float4 a = Src[clamp(i, 0, hi)];
    const float4 b = Src[clamp(i + int2(1, 0), 0, hi)];
    const float4 c = Src[clamp(i + int2(0, 1), 0, hi)];
    const float4 d = Src[clamp(i + int2(1, 1), 0, hi)];
    Dst[id.xy] = lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}
