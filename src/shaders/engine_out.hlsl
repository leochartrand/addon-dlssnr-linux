// The NR engine's output: its composed frame, float32 [3, field_h, field_w] -> the same region of the
// engine's raw answer atlas (engine_temporal turns it into the answer).
RWByteAddressBuffer Frame : register(u0);
RWTexture2D<float4> Dst : register(u1);

cbuffer EngineConstants : register(b0)
{
    uint g_x;
    uint g_y;
    uint g_w;
    uint g_h;
    uint g_field_w;
    uint g_field_h;
};

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_w || id.y >= g_h)
        return;
    uint plane = g_field_w * g_field_h;
    uint i = id.y * g_field_w + id.x;
    float3 c = float3(asfloat(Frame.Load(i * 4)), asfloat(Frame.Load((plane + i) * 4)),
                      asfloat(Frame.Load((2 * plane + i) * 4)));
    Dst[uint2(g_x + id.x, g_y + id.y)] = float4(saturate(c), 1.0);
}
