// Split mode, first thing each frame: place the crop on the GPU, from the newest gaze point the
// tracker thread wrote into a mapped buffer -- not the one there was when the frame was recorded,
// which with the host's submit-ahead is a frame older. The crop moves on a grid and holds until the
// gaze is 3/4 of a step away (fov.hpp), and the state keeps this frame's origin and the last one's
// for the motion vectors. Every split pass reads the origins back through CropOrigin().
StructuredBuffer<float> Gaze : register(t0);      // x, y normalised from the top-left
RWStructuredBuffer<float> State : register(u0);   // crop x, y; previous crop x, y (full pixels)

cbuffer CropConstants : register(b0)
{
    uint g_full_w;
    uint g_full_h;
    uint g_crop_w;
    uint g_crop_h;
    int g_snap;        // grid step, full pixels (<= 1: none)
    uint g_reset;      // first frame or a new layout: no hold, and no previous origin
    uint g_use_gaze;   // the tracker is live; otherwise g_centre
    float g_centre_x;
    float g_centre_y;
};

int Place(float centre, int size, int full, int prev)
{
    int o = (int)round(centre - 0.5 * (float)size);
    if (g_snap > 1)
    {
        if (prev >= 0 && abs(o - prev) <= g_snap * 3 / 4)
            o = prev;
        else
            o = (int)round((float)o / (float)g_snap) * g_snap;
    }
    return clamp(o, 0, full - size);
}

[numthreads(1, 1, 1)]
void main()
{
    float2 centre = float2(g_centre_x, g_centre_y);
    if (g_use_gaze != 0)
        centre = saturate(float2(Gaze[0], Gaze[1])) * float2(g_full_w, g_full_h);
    int2 prev = g_reset != 0 ? int2(-1, -1) : int2(State[0], State[1]);
    int2 o = int2(Place(centre.x, g_crop_w, g_full_w, prev.x), Place(centre.y, g_crop_h, g_full_h, prev.y));
    State[2] = g_reset != 0 ? (float)o.x : State[0];
    State[3] = g_reset != 0 ? (float)o.y : State[1];
    State[0] = (float)o.x;
    State[1] = (float)o.y;
}
