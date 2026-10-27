// Foveated NR: the frame is squeezed before the model -- 1:1 inside the fovea, blending down to
// the edge scale beyond it -- and the model's answer is stretched back after. The mapping is
// separable and lives in one float buffer the host rewrites every frame:
//
//   fwd[x]  working-space coordinate of full-resolution pixel centre x   (this frame)
//   prev[x] the same for the previous frame's mapping (motion vectors cross both)
//   inv[u]  full-resolution coordinate of working pixel centre u
//
// with the X table followed by the Y table in each block.
//
// Split foveation uses no tables: the model's working texture is an atlas of a full-resolution
// crop around the gaze (left, at the origin) and the whole frame at low resolution (from g_low_x),
// and the answer's broad tone is taken from the low-resolution half everywhere (fov_split_*).

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
    float g_centre_x;  // debug view only: the fovea centre and the raw gaze point, full pixels
    float g_centre_y;  // (gaze x < 0 = no gaze)
    float g_gaze_x;
    float g_gaze_y;
    uint g_crop_x;     // split: the crop's origin in full pixels, this frame and the last
    uint g_crop_y;
    int g_prev_crop_x;
    int g_prev_crop_y;
    uint g_crop_w;
    uint g_crop_h;
    uint g_low_x;      // split: where the low-resolution frame starts in the atlas, and its size
    uint g_low_w;
    uint g_low_h;
    float g_feather;   // split: blend from the crop's edge inwards, full pixels
    uint g_blur_axis;  // split blur: 0 horizontal, 1 vertical (and form the gains)
    float g_sigma;     // split blur: Gaussian sigma, low-resolution pixels
    float g_tone_alpha; // split blur V: weight of this frame's whole-frame offset against the
                        // reprojected history (1 = no history, 0 = history only)
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

// Mean of a full-resolution texture over a scale-sized footprint around a full-resolution
// position: four bilinear taps, each the mean of a 2x2 block (exact for a 4x scale).
float4 BoxSample(Texture2D<float4> src, float2 full, float2 scale)
{
    float2 inv = 1.0 / float2(g_full_w, g_full_h);
    float2 o = scale * 0.25;
    return 0.25 * (src.SampleLevel(Linear, (full + float2(-o.x, -o.y)) * inv, 0) +
                   src.SampleLevel(Linear, (full + float2(o.x, -o.y)) * inv, 0) +
                   src.SampleLevel(Linear, (full + float2(-o.x, o.y)) * inv, 0) +
                   src.SampleLevel(Linear, (full + float2(o.x, o.y)) * inv, 0));
}
