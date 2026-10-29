// Foveated NR: the model's working texture is an atlas of a full-resolution crop around the gaze
// (left, at the origin) and the whole frame at low resolution (from g_low_x); the answer's broad tone
// is taken from the low-resolution half everywhere (fov_split_*).

StructuredBuffer<float> CropState : register(t3);   // fov_split_crop: origin, previous origin
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
    uint g_reserved0;  // (the squeezed mode's table offsets; kept so the layout below stays put)
    uint g_reserved1;
    uint g_reserved2;
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
    float g_tone_alpha; // split blur V: least weight of this frame's whole-frame offset against
                        // the reprojected history (1 = no history, 0 = history only: no new pass)
    float g_tone_deadband; // split blur V: input change (display units) below which the tone holds
    uint g_pass_debug;     // split expand: 0 off, 1-3 the fovea pass alone, 4-6 the periphery pass alone
    float g_crop_tone;     // split expand: how much of the crop's broad tone is the periphery's (1) or its own (0)
};

// The crop's origin this frame and last, in full pixels, as fov_split_crop placed it on the GPU from
// the newest gaze.
float2 CropOrigin() { return float2(CropState[0], CropState[1]); }
float2 PrevCropOrigin() { return float2(CropState[2], CropState[3]); }

// Mean of a full-resolution texture over a scale-sized footprint around a full-resolution
// position: n x n bilinear taps per axis pair, each the mean of a 2x2 block, n = ceil(scale / 2)
// (up to 4: footprints to 8x). Fewer taps than the footprint needs would skip pixels, and the
// small picture would alias -- shimmer as the camera pans.
float4 BoxSample(Texture2D<float4> src, float2 full, float2 scale)
{
    float2 inv = 1.0 / float2(g_full_w, g_full_h);
    uint2 n = (uint2)clamp(ceil(scale * 0.5), 1.0, 4.0);
    float2 step = scale / (float2)n;               // tap spacing: the footprint in n equal cells
    float2 first = full - 0.5 * scale + 0.5 * step;
    float4 sum = 0.0;
    for (uint y = 0; y < n.y; ++y)
        for (uint x = 0; x < n.x; ++x)
            sum += src.SampleLevel(Linear, (first + step * float2(x, y)) * inv, 0);
    return sum / (float)(n.x * n.y);
}
