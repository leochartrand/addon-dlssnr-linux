// The NR engine's input: an atlas region (the display-referred proxy) -> float32 [3, field_h, field_w]
// at k/255, the region's last row and column mirrored out to the field (dlssnr frame_to_tensor).
Texture2D<float4> Src : register(t0);
RWByteAddressBuffer Image : register(u0);

cbuffer EngineConstants : register(b0)
{
    uint g_x;
    uint g_y;
    uint g_w;
    uint g_h;
    uint g_field_w;
    uint g_field_h;
};

// Reflect once about the last pixel, then clamp what is still outside.
uint Mirror(uint i, uint n)
{
    int r = i < n ? (int)i : 2 * (int)n - (int)i - 2;
    return (uint)clamp(r, 0, (int)n - 1);
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_field_w || id.y >= g_field_h)
        return;
    float3 c = saturate(Src[uint2(g_x + Mirror(id.x, g_w), g_y + Mirror(id.y, g_h))].rgb);
    c = floor(c * 255.0 + 0.5) / 255.0;
    uint plane = g_field_w * g_field_h;
    uint i = id.y * g_field_w + id.x;
    Image.Store(i * 4, asuint(c.r));
    Image.Store((plane + i) * 4, asuint(c.g));
    Image.Store((2 * plane + i) * 4, asuint(c.b));
}
