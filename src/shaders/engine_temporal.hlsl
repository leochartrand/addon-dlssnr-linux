// The NR engine's history. The engine sees one frame at a time (NVIDIA's network blends in its own
// previous output; ours has no such input), so a small change of the input can flip its answer from
// one frame to the next. This pass keeps what the model changed -- answer minus input -- reprojected
// with the atlas motion vectors (the crop's include its move) and blended with this frame's, clamped
// to this frame's 3x3 neighbourhood so a disocclusion or a real change is not smeared. The answer
// becomes input + the blended change; the change is also the next frame's history. The engine's
// own answers arrive in Raw (engine_out), so the neighbourhood reads never race the writes.
Texture2D<float4> Proxy : register(t0);      // the model's input atlas
Texture2D<float2> Motion : register(t1);     // atlas pixels: previous position = current + mv
Texture2D<float4> HistPrev : register(t2);   // last frame's change, atlas layout
Texture2D<float4> Raw : register(t3);        // the engine's answers, atlas layout
RWTexture2D<float4> Answer : register(u1);
RWTexture2D<float4> HistNext : register(u2);
SamplerState Linear : register(s0);

cbuffer EngineConstants : register(b0)
{
    uint g_x;          // unused here (the copy passes' region)
    uint g_y;
    uint g_w;
    uint g_h;
    uint g_field_w;
    uint g_field_h;
    uint g_crop_w;     // the atlas: the crop at the origin, the low-res frame from g_low_x
    uint g_crop_h;
    uint g_low_x;
    uint g_low_w;
    uint g_low_h;
    float g_alpha;     // weight of this frame's change (1 = no history)
    uint g_low_ran;    // the low-res region has a fresh answer this frame
    uint g_atlas_w;    // the history textures' size
    uint g_atlas_h;
};

float3 Change(int2 p, int2 lo, int2 hi)
{
    p = clamp(p, lo, hi);
    return Raw[p].rgb - Proxy[p].rgb;
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    int2 lo, hi;
    if (id.x < g_crop_w && id.y < g_crop_h)
    {
        lo = int2(0, 0);
        hi = int2(g_crop_w - 1, g_crop_h - 1);
    }
    else if (g_low_ran != 0 && id.x >= g_low_x && id.x < g_low_x + g_low_w && id.y < g_low_h)
    {
        lo = int2(g_low_x, 0);
        hi = int2(g_low_x + g_low_w - 1, g_low_h - 1);
    }
    else
        return;

    int2 p = int2(id.xy);
    float3 now = Change(p, lo, hi);
    float3 mn = now, mx = now;
    [unroll] for (int dy = -1; dy <= 1; ++dy)
        [unroll] for (int dx = -1; dx <= 1; ++dx)
        {
            float3 c = Change(p + int2(dx, dy), lo, hi);
            mn = min(mn, c);
            mx = max(mx, c);
        }

    float3 change = now;
    float2 prev = (float2)p + 0.5 + Motion[p];
    // History only from inside the same region (what the crop's move uncovered has none).
    if (g_alpha < 1.0 && all(prev >= (float2)lo + 0.5) && all(prev <= (float2)hi + 0.5))
    {
        float3 hist = HistPrev.SampleLevel(Linear, prev / float2(g_atlas_w, g_atlas_h), 0).rgb;
        change = lerp(clamp(hist, mn, mx), now, g_alpha);
    }
    Answer[p] = float4(saturate(Proxy[p].rgb + change), 1.0);
    HistNext[p] = float4(change, 1.0);
}
