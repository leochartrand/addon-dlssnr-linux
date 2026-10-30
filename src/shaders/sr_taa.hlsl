// Foveated super resolution, periphery (SrFovea=3): a small temporal accumulation at render resolution, so the
// game's jittered frame settles before it is stretched to the output. Last frame's accumulation is fetched where
// the motion vectors say the pixel was, clamped to the current 3x3 neighbourhood (no ghosts), and blended in.
Texture2D<float4> Color : register(t0);
Texture2D<float4> History : register(t1);
Texture2D<float2> Motion : register(t2);
RWTexture2D<float4> Out : register(u0);

cbuffer Constants : register(b0)
{
    uint g_w;
    uint g_h;
    float g_mv_x;      // the game's MV scale: motion * scale is in render pixels
    float g_mv_y;
    float g_current;   // the weight of this frame (1 on a reset)
    uint g_pad0;
    uint g_pad1;
    uint g_pad2;
};

float4 HistoryAt(float2 p)   // bilinear, p in pixels (centres at +0.5)
{
    const float2 q = p - 0.5;
    const int2 i = int2(floor(q));
    const float2 f = q - float2(i);
    const int2 hi = int2(g_w, g_h) - 1;
    const float4 a = History[clamp(i, 0, hi)];
    const float4 b = History[clamp(i + int2(1, 0), 0, hi)];
    const float4 c = History[clamp(i + int2(0, 1), 0, hi)];
    const float4 d = History[clamp(i + int2(1, 1), 0, hi)];
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_w || id.y >= g_h)
        return;
    const int2 hi = int2(g_w, g_h) - 1;
    const float4 cur = Color[id.xy];
    float4 lo = cur, up = cur;
    [unroll] for (int y = -1; y <= 1; ++y)
        [unroll] for (int x = -1; x <= 1; ++x)
        {
            const float4 n = Color[clamp(int2(id.xy) + int2(x, y), 0, hi)];
            lo = min(lo, n);
            up = max(up, n);
        }
    const float2 prev = float2(id.xy) + 0.5 + Motion[id.xy] * float2(g_mv_x, g_mv_y);
    const bool inside = all(prev >= 0.0) && all(prev <= float2(g_w, g_h));
    const float w = inside ? g_current : 1.0;
    const float4 h = clamp(HistoryAt(prev), lo, up);
    Out[id.xy] = lerp(h, cur, w);
}
