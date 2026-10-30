// Foveated super resolution: the crop's own DLSS-SR answer (its rectangle) blended into the frame, fully inside
// and fading out over the margin at its edges, so the two upscales meet without a seam.
Texture2D<float4> Crop : register(t0);
Texture2D<float4> Unused1 : register(t1);
Texture2D<float4> Unused2 : register(t2);
RWTexture2D<float4> Frame : register(u0);

cbuffer Constants : register(b0)
{
    uint g_base_x;   // the rectangle's origin in the frame
    uint g_base_y;
    uint g_w;        // its size
    uint g_h;
    float g_feather; // output pixels over which it fades in from each edge
    uint g_pad0;
    uint g_pad1;
    uint g_pad2;
};

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_w || id.y >= g_h)
        return;
    const float2 p = float2(id.xy) + 0.5;
    const float edge = min(min(p.x, g_w - p.x), min(p.y, g_h - p.y));
    const float w = g_feather > 0.0 ? smoothstep(0.0, 1.0, saturate(edge / g_feather)) : 1.0;
    const uint2 at = uint2(g_base_x, g_base_y) + id.xy;
    Frame[at] = lerp(Frame[at], Crop[id.xy], w);
}
