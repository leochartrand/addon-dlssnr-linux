// Foveated super resolution: how many frames of DLSS history each pixel of the crop's rectangle has. The rectangle
// moved by (g_dx, g_dy) output pixels since the frame before; a pixel that was inside it then carries its age over
// (one more frame), one that just came in starts at 0, and a reset starts every pixel at 0.
Texture2D<float> Old : register(t0);
Texture2D<float4> Unused1 : register(t1);
Texture2D<float4> Unused2 : register(t2);
RWTexture2D<float> Age : register(u0);

cbuffer Constants : register(b0)
{
    uint g_w;        // the rectangle's size
    uint g_h;
    int g_dx;        // its move since the frame before
    int g_dy;
    uint g_reset;    // 1: no pixel has history
    uint g_pad0;
    uint g_pad1;
    uint g_pad2;
};

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_w || id.y >= g_h)
        return;
    const int2 was = int2(id.xy) + int2(g_dx, g_dy);
    const bool inside = was.x >= 0 && was.y >= 0 && was.x < (int)g_w && was.y < (int)g_h;
    Age[id.xy] = (g_reset == 0 && inside) ? min(Old[uint2(was)] + 1.0, 255.0) : 0.0;
}
