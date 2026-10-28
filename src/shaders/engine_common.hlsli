// The NR engine's passes (engine_in, engine_out): NVIDIA's temporal loop, as OpenDLSS-NR documents it
// (docs/frame.md). The network's features 7-9 are the previous output reprojected (or this frame where
// there is no history), and its fourth head channel is a per-pixel logit for how much of that history
// to keep:
//
//   neural   = the engine's composed frame (clamp(proxy + rgb / 4))
//   weight   = clamp(sigmoid(logit) * blend_scale, 0, 1)        only where there is a history
//   answer   = lerp(neural, history, weight)
//   history' = answer, truncated toward zero to the half grid
//
// The history is sampled at the reprojected position with a five-tap Catmull-Rom filter, clamped to
// the region it came from. Each region of the atlas (the crop, the low-res frame) keeps its own.

Texture2D<float4> Proxy : register(t0);      // the model's input atlas (display-referred)
Texture2D<float2> Motion : register(t1);     // atlas pixels: previous position = current + mv
Texture2D<float4> HistPrev : register(t2);   // last frame's answers, atlas layout
SamplerState Linear : register(s0);

cbuffer EngineConstants : register(b0)
{
    uint g_x;          // the region in the atlas
    uint g_y;
    uint g_w;
    uint g_h;
    uint g_field_w;    // the engine's field (the region, mirrored out to it)
    uint g_field_h;
    uint g_atlas_w;
    uint g_atlas_h;
    uint g_valid;      // the region has a history (last frame ran it, no reset)
    float g_blend;     // the learned cap on the blend, times the user's history strength (0 = none)
    float g_motion;    // scale on the motion the history is reprojected along (1 = as given, 0 = none)
};

// Reflect once about the last pixel, then clamp what is still outside.
uint Mirror(uint i, uint n)
{
    int r = i < n ? (int)i : 2 * (int)n - (int)i - 2;
    return (uint)clamp(r, 0, (int)n - 1);
}

// Where the atlas pixel p was last frame, and whether the region has it.
bool Reproject(uint2 p, out float2 prev)
{
    prev = (float2)p + 0.5 + Motion[p] * g_motion;
    float2 lo = float2(g_x, g_y) + 0.5, hi = float2(g_x + g_w, g_y + g_h) - 0.5;
    return g_valid != 0 && g_blend > 0.0 && all(prev >= lo) && all(prev <= hi);
}

// Five-tap Catmull-Rom (the four axis taps and the centre, with the bilinear-weight trick), each tap
// clamped to the region.
float3 History(float2 pos)
{
    float2 lo = float2(g_x, g_y) + 0.5, hi = float2(g_x + g_w, g_y + g_h) - 0.5;
    float2 inv = 1.0 / float2(g_atlas_w, g_atlas_h);
    float2 centre = floor(pos - 0.5) + 0.5;
    float2 f = pos - centre;
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);
    float2 w12 = w1 + w2;
    float2 t0 = clamp(centre - 1.0, lo, hi) * inv;
    float2 t3 = clamp(centre + 2.0, lo, hi) * inv;
    float2 t12 = clamp(centre + w2 / w12, lo, hi) * inv;
    float4 c = HistPrev.SampleLevel(Linear, float2(t12.x, t0.y), 0) * (w12.x * w0.y) +
               HistPrev.SampleLevel(Linear, float2(t0.x, t12.y), 0) * (w0.x * w12.y) +
               HistPrev.SampleLevel(Linear, float2(t12.x, t12.y), 0) * (w12.x * w12.y) +
               HistPrev.SampleLevel(Linear, float2(t3.x, t12.y), 0) * (w3.x * w12.y) +
               HistPrev.SampleLevel(Linear, float2(t12.x, t3.y), 0) * (w12.x * w3.y);
    float norm = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return saturate(c.rgb / norm);
}
