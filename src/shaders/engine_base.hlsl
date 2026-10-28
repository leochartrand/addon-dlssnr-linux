// The NR engine's per-pixel input block for one frame, as NVIDIA's library builds it: the lift (the first
// layer's 32x16 matrix) of three noise features, a constant and the five conditioning values (style, tone,
// structure, skin, other), fp16, [32, height, width] channels-last. The noise is renewed every frame (the
// library counts frames into its seed) and the conditioning carries the live controls; the colour features
// are not part of the block (the program adds them).
//
// The noise is dlssnr-pytorch's gaussian_noise: per pixel, a PCG-style hash of (x, y, seed) into four
// uniform numbers and two Box-Muller transforms.

ByteAddressBuffer Lift : register(t0);     // [32][16] float32
RWByteAddressBuffer Base : register(u0);   // the program's block, at its offset

cbuffer BaseConstants : register(b0)
{
    uint g_w;
    uint g_h;
    uint g_seed;
    float g_noise_gain;
    float g_constant;
    float g_style;      // the five conditioning values, already as the library forms them
    float g_tone;
    float g_structure;
    float g_skin;
    float g_other;
};

uint Pcg(uint v)
{
    v = (v >> ((v >> 28) + 4)) ^ v;
    return v * 0x108EF2D9u;
}

// Uniform in (0, 1] from a 32-bit stream value.
float Unit(uint s)
{
    uint t = Pcg(s);
    return (float)(((t >> 30) ^ (t >> 8)) + 1u) * exp2(-24.0);
}

float Half(float x) { return f16tof32(f32tof16(x)); }

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_w || id.y >= g_h)
        return;
    uint h0 = Pcg((id.x * 0x8DA6B343u) ^ (g_seed * 0x9E3779B9u) ^ (id.y * 0xD8163841u) ^ 0x243F6A88u);
    uint hh = (h0 >> 22) ^ h0;
    float ua = Unit(hh * 0x2C9277B5u + 0xAC564B05u), ub = Unit(hh * 0xFA6DC5F9u + 0x4712A88Eu);
    float uc = Unit(hh * 0xCAA5B80Du + 0x21DD796Bu), ud = Unit(hh * 0x83232C31u + 0x3463E0ACu);
    float ra = sqrt(-2.0 * log(ua)), rc = sqrt(-2.0 * log(uc));
    const float two_pi = 6.28318530718;

    float f[16];
    f[0] = Half(g_noise_gain * ra * cos(ub * two_pi));
    f[1] = Half(g_noise_gain * ra * sin(ub * two_pi));
    f[2] = Half(g_noise_gain * rc * cos(ud * two_pi));
    f[3] = g_constant;
    [unroll] for (int k = 4; k < 10; ++k) f[k] = 0.0;   // the colour features: added by the program
    f[10] = Half(g_style);
    f[11] = Half(g_tone);
    f[12] = Half(g_structure);
    f[13] = Half(g_skin);
    f[14] = Half(g_other);
    f[15] = 0.0;

    uint at = (id.y * g_w + id.x) * 64;   // 32 halves per pixel
    [unroll] for (uint q = 0; q < 4; ++q)
    {
        uint4 packed;
        [unroll] for (uint j = 0; j < 4; ++j)
        {
            uint c = q * 8 + j * 2;
            float a = 0.0, b = 0.0;
            [unroll] for (int k = 0; k < 16; ++k)
            {
                a += asfloat(Lift.Load((c * 16 + k) * 4)) * f[k];
                b += asfloat(Lift.Load(((c + 1) * 16 + k) * 4)) * f[k];
            }
            packed[j] = f32tof16(a) | (f32tof16(b) << 16);
        }
        Base.Store4(at + q * 16, packed);
    }
}
