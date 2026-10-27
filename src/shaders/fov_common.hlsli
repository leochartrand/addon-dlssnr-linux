// Foveated NR: the frame is squeezed before the model -- 1:1 inside the fovea, blending down to
// the edge scale beyond it -- and the model's answer is stretched back after. The mapping is
// separable and lives in one float buffer the host rewrites every frame:
//
//   fwd[x]  working-space coordinate of full-resolution pixel centre x   (this frame)
//   prev[x] the same for the previous frame's mapping (motion vectors cross both)
//   inv[u]  full-resolution coordinate of working pixel centre u
//
// with the X table followed by the Y table in each block.

StructuredBuffer<float> Lut : register(t3);
SamplerState Linear : register(s0);

cbuffer FovConstants : register(b0)
{
    uint g_full_w;
    uint g_full_h;
    uint g_work_w;
    uint g_work_h;
    uint g_render_w;   // the depth and motion-vector textures' resolution
    uint g_render_h;
    float g_mv_scale_x;
    float g_mv_scale_y;
    uint g_fwd;        // offsets into Lut, in floats
    uint g_prev;
    uint g_inv;
};

// Continuous lookup into a per-pixel-centre table: pos is a coordinate in that table's space
// (pixel centres at i + 0.5), clamped to its ends.
float LutAt(uint base, uint count, float pos)
{
    float i = clamp(pos - 0.5, 0.0, (float)(count - 1));
    uint i0 = (uint)i;
    uint i1 = min(i0 + 1, count - 1);
    return lerp(Lut[base + i0], Lut[base + i1], i - (float)i0);
}

float2 Forward(uint base, float2 full_pos)
{
    return float2(LutAt(base, g_full_w, full_pos.x), LutAt(base + g_full_w, g_full_h, full_pos.y));
}

float2 Inverse(uint2 work_px)
{
    return float2(Lut[g_inv + work_px.x], Lut[g_inv + g_work_w + work_px.y]);
}
