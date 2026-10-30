// Foveated super resolution: the crop's motion vectors, read from the game's at the rectangle's render origin and
// shifted by how far the rectangle moved since the frame before, so the crop's DLSS-SR carries its history over
// instead of starting again (its pixels moved by exactly that much in its own coordinates).
Texture2D<float4> Motion : register(t0);
Texture2D<float4> Unused1 : register(t1);
Texture2D<float4> Unused2 : register(t2);
RWTexture2D<float2> Shifted : register(u0);

cbuffer Constants : register(b0)
{
    uint g_base_x;   // the rectangle's origin in the game's motion vectors (render pixels)
    uint g_base_y;
    uint g_w;        // its size
    uint g_h;
    float g_add_x;   // the move, in the motion vectors' units (render pixels / MV scale)
    float g_add_y;
    uint g_pad0;
    uint g_pad1;
};

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_w || id.y >= g_h)
        return;
    Shifted[id.xy] = Motion[uint2(g_base_x, g_base_y) + id.xy].xy + float2(g_add_x, g_add_y);
}
