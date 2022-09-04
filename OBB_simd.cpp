#include "OBB_simd.h"
#include <cassert>
#include <limits>
#include <tuple>

#ifdef _MSC_VER
#    define OBB_ALIGN16 __declspec(align(16))
#    define OBB_ALIGN(x) __declspec(align(x))
#else
#    define OBB_ALIGN16 __attribute__((aligned(16)))
#    define OBB_ALIGN(x) __attribute__((aligned(x)))
#endif

#if defined(__SSE2__) || defined(__SSE3__) || defined(__SSE4_1__) || defined(__SSE4_2__) || defined(__AVX__) || defined(__AVX2__)
#ifdef _MSC_VER
#    include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
# include <x86intrin.h>
#endif
using float4 = __m128;
using int4 = __m128i;
#define OBB_SSE (1)
#define OBB_NEON (0)

#define SIMD_LOAD_F4(x) _mm_load_ps(x)
#define SIMD_STORE_F4(x0, x1) _mm_store_ps(x0, x1)

#define SIMD_SET1_F4(x) _mm_set1_ps(x)
#define SIMD_SET1_I4(x) _mm_set1_epi32(x)

inline float4 SIMD_SET_F4(obb::f32 x, obb::f32 y, obb::f32 z, obb::f32 w)
{
    return _mm_set_ps(x, y, z, w);
}

inline float4 SIMD_SETR_F4(obb::f32 x, obb::f32 y, obb::f32 z, obb::f32 w)
{
    return _mm_setr_ps(x, y, z, w);
}

inline int4 SIMD_SET_I4(obb::s32 x, obb::s32 y, obb::s32 z, obb::s32 w)
{
    return _mm_set_epi32(x, y, z, w);
}

#define SIMD_CAST_F4_I4(x) _mm_castps_si128(x)
#define SIMD_CAST_I4_F4(x) _mm_castsi128_ps(x)

#define SIMD_ADD_F4(x0, x1) _mm_add_ps(x0, x1)
#define SIMD_SUB_F4(x0, x1) _mm_sub_ps(x0, x1)
#define SIMD_MUL_F4(x0, x1) _mm_mul_ps(x0, x1)
#define SIMD_MIN_F4(x0, x1) _mm_min_ps(x0, x1)
#define SIMD_MAX_F4(x0, x1) _mm_max_ps(x0, x1)


#define SIMD_CMPLT_F4(x0, x1) _mm_cmplt_ps(x0, x1)
#define SIMD_AND_F4(x0, x1) _mm_and_ps(x0, x1)
#define SIMD_OR_F4(x0, x1) _mm_or_ps(x0, x1)
#define SIMD_ANDNOT_F4(x0, x1) _mm_andnot_ps(x0, x1)

#define SIMD_STORE_I4(x0, x1) _mm_store_si128(reinterpret_cast<int4*>(x0), x1)

#define SIMD_ADD_I4(x0, x1) _mm_add_epi32(x0, x1)

#define SIMD_AND_I4(x0, x1) _mm_and_si128(x0, x1)
#define SIMD_OR_I4(x0, x1) _mm_or_si128(x0, x1)
#define SIMD_ANDNOT_I4(x0, x1) _mm_andnot_si128(x0, x1)

#define blend(x0, x1, c0, c1, c2, c3) \
    _mm_blend_ps(x0, x1, (c0 << 0) | (c1 << 1) | (c2 << 2) | (c3 << 3))

#define shuffle(x0, x1, c0, c1, c2, c3) \
    _mm_shuffle_ps(x0, x1, _MM_SHUFFLE(c3, c2, c1, c0))

#elif defined(__ARM_NEON)
#    include <arm_neon.h>
using float4 = float32x4_t;
using int4 = int32x4_t;
#define OBB_SSE (0)
#define OBB_NEON (1)

#define SIMD_LOAD_F4(x) vld1q_f32(x)
#define SIMD_STORE_F4(x0, x1) vst1q_f32(x0, x1)

#define SIMD_SET1_F4(x) vdupq_n_f32(x)
#define SIMD_SET1_I4(x) vmovq_n_s32(x)

inline float4 SIMD_SET_F4(obb::f32 x, obb::f32 y, obb::f32 z, obb::f32 w)
{
	obb::f32 OBB_ALIGN16 data[4] = {w, z, y, x};
	return vld1q_f32(data);
}

inline float4 SIMD_SETR_F4(obb::f32 x, obb::f32 y, obb::f32 z, obb::f32 w)
{
	obb::f32 OBB_ALIGN16 data[4] = {x, y, z, w};
	return vld1q_f32(data);
}

inline int4 SIMD_SET_I4(obb::s32 x, obb::s32 y, obb::s32 z, obb::s32 w)
{
	obb::s32 OBB_ALIGN16 data[4] = {w, z, y, x};
	return vld1q_s32(data);
}

#define SIMD_CAST_F4_I4(x) vreinterpretq_s32_f32(x)
#define SIMD_CAST_I4_F4(x) vreinterpretq_f32_s32(x)

#define SIMD_ADD_F4(x0, x1) vaddq_f32(x0, x1)
#define SIMD_SUB_F4(x0, x1) vsubq_f32(x0, x1)
#define SIMD_MUL_F4(x0, x1) vmulq_f32(x0, x1)
#define SIMD_MIN_F4(x0, x1) vminq_f32(x0, x1)
#define SIMD_MAX_F4(x0, x1) vmaxq_f32(x0, x1)

#define SIMD_CMPLT_F4(x0, x1) vreinterpretq_f32_u32(vcltq_f32(x0, x1))
#define SIMD_AND_F4(x0, x1) SIMD_CAST_I4_F4(vandq_s32(SIMD_CAST_F4_I4(x0), SIMD_CAST_F4_I4(x1)))
#define SIMD_OR_F4(x0, x1) SIMD_CAST_I4_F4(vorrq_s32(SIMD_CAST_F4_I4(x0), SIMD_CAST_F4_I4(x1)))
#define SIMD_ANDNOT_F4(x0, x1) SIMD_CAST_I4_F4(vbicq_s32(SIMD_CAST_F4_I4(x0), SIMD_CAST_F4_I4(x1)))

#define SIMD_STORE_I4(x0, x1) vst1q_s32(reinterpret_cast<obb::s32*>(x0), x1)

#define SIMD_ADD_I4(x0, x1) vaddq_s32(x0, x1)

#define SIMD_AND_I4(x0, x1) vandq_s32(x0, x1)
#define SIMD_OR_I4(x0, x1) vorrq_s32(x0, x1)
#define SIMD_ANDNOT_I4(x0, x1) vbicq_s32(x0, x1)

namespace
{
float4 shuffle(float4 x0, float4 x1, obb::u32 c0, obb::u32 c1, obb::u32 c2, obb::u32 c3)
{
    assert(c0<4 && c1<4 && c2<4 && c3<4);
    obb::u32 imm = ((c3 << 6) | (c2 << 4) | (c1 << 2) | (c0));
	float4 ret;
	ret[0] = x0[imm & 0x3];
	ret[1] = x0[(imm >> 2) & 0x3];
	ret[2] = x1[(imm >> 4) & 0x03];
	ret[3] = x1[(imm >> 6) & 0x03];
	return ret;
}
}

#else
#define OBB_SSE (0)
#define OBB_NEON (0)
#endif

namespace obb
{
namespace
{
    static constexpr u32 Normals = 8U;
    static constexpr u32 InitialPoints = 16U;
    static constexpr f32 Sqrt2 = static_cast<f32>(1.4142135623730950488016887242097);
    static constexpr f32 Sqrt3 = static_cast<f32>(0.57735026918962576450914878050196);
    static constexpr f32 Infinity = std::numeric_limits<f32>::infinity();
    static constexpr u32 Dim = 4;
    static constexpr u32 Shift = 2;

#ifdef __cplusplus
#    if 201103L <= __cplusplus || 1900 <= _MSC_VER
#        define OBB_CPP11 1
#    endif
#endif

#ifndef OBB_NULL
#    ifdef __cplusplus
#        ifdef OBB_CPP11
#            define OBB_NULL nullptr
#        endif
#    else
#        define OBB_NULL ((void*)0)
#    endif
#endif

#define OBB_MALLOC(size) (::malloc(size))
#define OBB_FREE(mem) \
    ::free(mem); \
    (mem) = OBB_NULL

#if defined(_MSC_VER)
#    define OBB_ALIGNED_MALLOC(size) (_aligned_malloc(size, 16U))
#    define OBB_ALIGNED_FREE(mem) \
        _aligned_free(mem); \
        (mem) = OBB_NULL

#else
#    if 200112 <= _POSIX_C_SOURCE
    inline void* OBB_ALIGNED_MALLOC(size_t size)
    {
        void* ptr;
        return 0 == posix_memalign(&ptr, 16U, size) ? ptr : OBB_NULL;
    }
#    elif defined(_ISOC11_SOURCE)
    inline void* OBB_ALIGNED_MALLOC(size_t size)
    {
        size = (size + 15ULL) & ~15ULL;
        return aligned_alloc(16ULL, size);
    }
#    endif

#    define OBB_ALIGNED_FREE(mem) \
        ::free(mem); \
        (mem) = OBB_NULL
#endif

    template<u32 N>
    struct TSoA
    {
        static constexpr u32 size = N;
        f32 x_[N];
        f32 y_[N];
        f32 z_[N];
    };

    static const OBB_ALIGN16 Vector3 N08[Normals] = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {Sqrt3, Sqrt3, Sqrt3},
        {Sqrt3, Sqrt3, -Sqrt3},
        {Sqrt3, -Sqrt3, Sqrt3},
        {Sqrt3, -Sqrt3, -Sqrt3},
        //{Sqrt2, Sqrt2, 0.0f},
        {1.0f, 0.0f, 0.0f},
    };

    using SoA8 = TSoA<Normals * 2>;

#if 0
    void aos2soa(float4& x, float4& y, float4& z, const Vector3 src[4])
    {
        Vector3 tp0 = src[0];
        Vector3 tp1 = src[1];
        Vector3 tp2 = src[2];
        Vector3 tp3 = src[3];

        float4 t[3];
        t[0] = _mm_loadu_ps(reinterpret_cast<const f32*>(&src[0].x_)); // x0 y0 z0 x1
        t[1] = _mm_loadu_ps(reinterpret_cast<const f32*>(&src[1].y_)); // y1 z1 x2 y2
        t[2] = _mm_loadu_ps(reinterpret_cast<const f32*>(&src[2].z_)); // z2 x3 y3 z3
        x = blend(t[0], t[1], 0, 0, 1, 0);                             // x0 y0 x2 x1
        y = blend(t[0], t[1], 1, 0, 0, 1);                             // y1 y0 z0 y2
        z = blend(t[0], t[2], 1, 0, 0, 1);                             // z2 y0 z0 z3

        x = blend(x, t[2], 0, 1, 0, 0); // x0 x3 x2 x1
        y = blend(y, t[2], 0, 0, 1, 0); // y1 y0 y3 y2
        z = blend(z, t[1], 0, 1, 0, 0); // z2 z1 z0 z3

        x = shuffle(x, x, 0, 3, 2, 1);
        y = shuffle(y, y, 1, 0, 3, 2);
        z = shuffle(z, z, 2, 1, 0, 3);
    }
#endif

    f32 halfBoxArea(const Vector3& extent)
    {
        return extent.x_ * extent.y_ + extent.y_ * extent.z_ + extent.z_ * extent.x_;
    }

    f32 findMinMax(f32& minx, f32& maxx, const Vector3& normal, u32 size, const Vector3* points)
    {
#if 0
        minx = maxx = dot(points[0], normal);
        for(u32 i = 1; i < size; ++i) {
            f32 x = dot(points[i], normal);
            if(x < minx) {
                minx = x;
            }
            if(maxx < x) {
                maxx = x;
            }
        }
        return maxx - minx;
#else
        float4 nx = SIMD_SET1_F4(normal.x_);
        float4 ny = SIMD_SET1_F4(normal.y_);
        float4 nz = SIMD_SET1_F4(normal.z_);
        float4 minx4 = SIMD_SET1_F4(Infinity);
        float4 maxx4 = SIMD_SET1_F4(-Infinity);
        u32 size4 = size & ~(Dim - 1);
        for(u32 i = 0; i < size4; i += 4) {
            //float4 px, py, pz;
            //aos2soa(px, py, pz, points + i);
            float4 px = SIMD_SET_F4(points[i+3].x_, points[i+2].x_, points[i+1].x_, points[i+0].x_);
            float4 py = SIMD_SET_F4(points[i+3].y_, points[i+2].y_, points[i+1].y_, points[i+0].y_);
            float4 pz = SIMD_SET_F4(points[i+3].z_, points[i+2].z_, points[i+1].z_, points[i+0].z_);

            px = SIMD_MUL_F4(px, nx);
            py = SIMD_MUL_F4(py, ny);
            pz = SIMD_MUL_F4(pz, nz);
            float4 d = SIMD_ADD_F4(px, SIMD_ADD_F4(py, pz));
            minx4 = SIMD_MIN_F4(minx4, d);
            maxx4 = SIMD_MAX_F4(maxx4, d);
        }
        minx4 = SIMD_MIN_F4(minx4, shuffle(minx4, minx4, 1, 0, 3, 2));
        minx4 = SIMD_MIN_F4(minx4, shuffle(minx4, minx4, 2, 3, 0, 1));
        maxx4 = SIMD_MAX_F4(maxx4, shuffle(maxx4, maxx4, 1, 0, 3, 2));
        maxx4 = SIMD_MAX_F4(maxx4, shuffle(maxx4, maxx4, 2, 3, 0, 1));

        OBB_ALIGN16 f32 dmin[Dim];
        OBB_ALIGN16 f32 dmax[Dim];
        SIMD_STORE_F4(dmin, minx4);
        SIMD_STORE_F4(dmax, maxx4);

        minx = dmin[0];
        maxx = dmax[0];

        for(u32 i = size4; i < size; ++i) {
            f32 x = dot(points[i], normal);
            if(x < minx) {
                minx = x;
            }
            if(maxx < x) {
                maxx = x;
            }
        }

        return maxx - minx;
#endif
    }

    f32 findMinMax(f32& minx, f32& maxx, const Vector3& normal, const SoA8& soa)
    {
#if 0
        Vector3 p = {soa.x_[0], soa.y_[0], soa.z_[0]};
        minx = maxx = dot(p, normal);
        for(u32 i = 1; i < soa.size; ++i) {
            p = {soa.x_[i], soa.y_[i], soa.z_[i]};
            f32 x = dot(p, normal);
            if(x < minx) {
                minx = x;
            }
            if(maxx < x) {
                maxx = x;
            }
        }
        return maxx - minx;
#else

        float4 nx = SIMD_SET1_F4(normal.x_);
        float4 ny = SIMD_SET1_F4(normal.y_);
        float4 nz = SIMD_SET1_F4(normal.z_);
        float4 minx4;
        float4 maxx4;

        {
            float4 px = SIMD_LOAD_F4(soa.x_);
            float4 py = SIMD_LOAD_F4(soa.y_);
            float4 pz = SIMD_LOAD_F4(soa.z_);
            px = SIMD_MUL_F4(px, nx);
            py = SIMD_MUL_F4(py, ny);
            pz = SIMD_MUL_F4(pz, nz);
            minx4 = maxx4 = SIMD_ADD_F4(px, SIMD_ADD_F4(py, pz));
        }
        for(u32 i = Dim; i < InitialPoints; i += Dim) {
            float4 px = SIMD_LOAD_F4(soa.x_ + i);
            float4 py = SIMD_LOAD_F4(soa.y_ + i);
            float4 pz = SIMD_LOAD_F4(soa.z_ + i);
            px = SIMD_MUL_F4(px, nx);
            py = SIMD_MUL_F4(py, ny);
            pz = SIMD_MUL_F4(pz, nz);
            float4 d = SIMD_ADD_F4(px, SIMD_ADD_F4(py, pz));
            minx4 = SIMD_MIN_F4(minx4, d);
            maxx4 = SIMD_MAX_F4(maxx4, d);
        }
        minx4 = SIMD_MIN_F4(minx4, shuffle(minx4, minx4, 1, 0, 3, 2));
        minx4 = SIMD_MIN_F4(minx4, shuffle(minx4, minx4, 2, 3, 0, 1));
        maxx4 = SIMD_MAX_F4(maxx4, shuffle(maxx4, maxx4, 1, 0, 3, 2));
        maxx4 = SIMD_MAX_F4(maxx4, shuffle(maxx4, maxx4, 2, 3, 0, 1));

        OBB_ALIGN16 f32 dmin[Dim];
        OBB_ALIGN16 f32 dmax[Dim];
        SIMD_STORE_F4(dmin, minx4);
        SIMD_STORE_F4(dmax, maxx4);

        minx = dmin[0];
        maxx = dmax[0];
        return maxx - minx;
#endif
    }

    std::tuple<f32, u32, u32> findPoints(SoA8& results, u32 size, const Vector3* OBB_RESTRICT points)
    {
#if 0
        f32 maxDistance = 0.0f;
        u32 pair0;
        u32 pair1;
        for(u32 n = 0; n < Normals; ++n) {
            f32 minx = std::numeric_limits<f32>::infinity();
            f32 maxx = -std::numeric_limits<f32>::infinity();
            u32 mini = Invalid;
            u32 maxi = Invalid;
            const Vector3& N = N08[n];
            for(u32 i = 0; i < size; ++i) {
                f32 d = dot(N, points[i]);
                if(d < minx) {
                    minx = d;
                    mini = i;
                }
                if(maxx < d) {
                    maxx = d;
                    maxi = i;
                }
            }
            assert(Invalid != mini && Invalid != maxi);
            f32 distance = maxx - minx;
            if(maxDistance < distance) {
                maxDistance = distance;
                pair0 = mini;
                pair1 = maxi;
            }
            u32 i0 = n * 2 + 0;
            u32 i1 = n * 2 + 1;
            results.x_[i0] = points[mini].x_;
            results.x_[i1] = points[maxi].x_;
            results.y_[i0] = points[mini].y_;
            results.y_[i1] = points[maxi].y_;
            results.z_[i0] = points[mini].z_;
            results.z_[i1] = points[maxi].z_;
        }
        return std::make_tuple(maxDistance, pair0, pair1);
#else
        f32 minx[Normals];
        f32 maxx[Normals];
        u32 mini[Normals];
        u32 maxi[Normals];
        for(u32 i = 0; i < Normals; ++i) {
            minx[i] = std::numeric_limits<f32>::infinity();
            maxx[i] = -std::numeric_limits<f32>::infinity();
            mini[i] = Invalid;
            maxi[i] = Invalid;
        }
        float4 NX[2];
        float4 NY[2];
        float4 NZ[2];
        NX[0] = SIMD_SETR_F4(N08[0].x_, N08[1].x_, N08[2].x_, N08[3].x_);
        NX[1] = SIMD_SETR_F4(N08[4].x_, N08[5].x_, N08[6].x_, N08[7].x_);
        NY[0] = SIMD_SETR_F4(N08[0].y_, N08[1].y_, N08[2].y_, N08[3].y_);
        NY[1] = SIMD_SETR_F4(N08[4].y_, N08[5].y_, N08[6].y_, N08[7].y_);
        NZ[0] = SIMD_SETR_F4(N08[0].z_, N08[1].z_, N08[2].z_, N08[3].z_);
        NZ[1] = SIMD_SETR_F4(N08[4].z_, N08[5].z_, N08[6].z_, N08[7].z_);

        for(u32 i = 0; i < size; ++i) {
            float4 px = SIMD_SET1_F4(points[i].x_);
            float4 py = SIMD_SET1_F4(points[i].y_);
            float4 pz = SIMD_SET1_F4(points[i].z_);
            float4 d0 = SIMD_ADD_F4(SIMD_MUL_F4(NX[0], px), SIMD_ADD_F4(SIMD_MUL_F4(NY[0], py), SIMD_MUL_F4(NZ[0], pz)));
            float4 d1 = SIMD_ADD_F4(SIMD_MUL_F4(NX[1], px), SIMD_ADD_F4(SIMD_MUL_F4(NY[1], py), SIMD_MUL_F4(NZ[1], pz)));

            OBB_ALIGN16 f32 distances[Normals];
            SIMD_STORE_F4(distances, d0);
            SIMD_STORE_F4(distances + Dim, d1);

            for(u32 n = 0; n < Normals; ++n) {
                f32 d = distances[n];
                if(d < minx[n]) {
                    minx[n] = d;
                    mini[n] = i;
                }
                if(maxx[n] < d) {
                    maxx[n] = d;
                    maxi[n] = i;
                }
                assert(Invalid != mini[n] && Invalid != maxi[n]);
            }
        }
        f32 maxDistance = 0.0f;
        u32 pair0 = Invalid;
        u32 pair1 = Invalid;
        for(u32 n = 0; n < Normals; ++n) {
            f32 distance = maxx[n] - minx[n];
            if(maxDistance < distance) {
                maxDistance = distance;
                pair0 = mini[n];
                pair1 = maxi[n];
            }
            u32 i0 = n * 2 + 0;
            u32 i1 = n * 2 + 1;
            results.x_[i0] = points[mini[n]].x_;
            results.x_[i1] = points[maxi[n]].x_;
            results.y_[i0] = points[mini[n]].y_;
            results.y_[i1] = points[maxi[n]].y_;
            results.z_[i0] = points[mini[n]].z_;
            results.z_[i1] = points[maxi[n]].z_;
        }
        return std::make_tuple(maxDistance, pair0, pair1);
#endif
    }

    f32 distanceSqr(const Vector3& origin, const Vector3& direction, const Vector3& point)
    {
        Vector3 diff = origin - point;
        f32 d = dot(diff, direction);
        Vector3 p = diff - d * direction;
        return p.lengthSqr();
    }

    float4 distanceSqr(const float4 origin[3], const float4 direction[3], const float4 point[3])
    {
        float4 dx = SIMD_SUB_F4(origin[0], point[0]);
        float4 dy = SIMD_SUB_F4(origin[1], point[1]);
        float4 dz = SIMD_SUB_F4(origin[2], point[2]);

        float4 d = SIMD_ADD_F4(SIMD_MUL_F4(dx, direction[0]), SIMD_ADD_F4(SIMD_MUL_F4(dy, direction[1]), SIMD_MUL_F4(dz, direction[2])));
        float4 px = SIMD_SUB_F4(dx, SIMD_MUL_F4(d, direction[0]));
        float4 py = SIMD_SUB_F4(dy, SIMD_MUL_F4(d, direction[1]));
        float4 pz = SIMD_SUB_F4(dz, SIMD_MUL_F4(d, direction[2]));
        px = SIMD_MUL_F4(px, px);
        py = SIMD_MUL_F4(py, py);
        pz = SIMD_MUL_F4(pz, pz);
        return SIMD_ADD_F4(px, SIMD_ADD_F4(py, pz));
    }

    std::tuple<f32, u32> findFurthestPoint(const Vector3& origin, const Vector3& direction, u32 size, const Vector3* points)
    {
#if 0
        f32 maxDistance = distanceSqr(origin, direction, points[0]);
        u32 maxi = 0;
        for(u32 i = 1; i < size; ++i) {
            f32 distance = distanceSqr(origin, direction, points[i]);
            if(maxDistance < distance) {
                maxDistance = distance;
                maxi = i;
            }
        }
        return std::make_tuple(maxDistance, maxi);
#else
        float4 d[3];
        d[0] = SIMD_SET1_F4(direction.x_);
        d[1] = SIMD_SET1_F4(direction.y_);
        d[2] = SIMD_SET1_F4(direction.z_);
        float4 o[3];
        o[0] = SIMD_SET1_F4(origin.x_);
        o[1] = SIMD_SET1_F4(origin.y_);
        o[2] = SIMD_SET1_F4(origin.z_);
        float4 maxx4 = SIMD_SET1_F4(-Infinity);
        int4 maxi4 = SIMD_SET1_I4(-1);
        int4 index4 = SIMD_SET_I4(3, 2, 1, 0);
        int4 four = SIMD_SET1_I4(4);
        u32 size4 = size & ~(Dim - 1);
        for(u32 i = 0; i < size4; i += 4) {
            float4 p[3];
            //aos2soa(p[0], p[1], p[2], points + i);
            p[0] = SIMD_SET_F4(points[i+3].x_, points[i+2].x_, points[i+1].x_, points[i+0].x_);
            p[1] = SIMD_SET_F4(points[i+3].y_, points[i+2].y_, points[i+1].y_, points[i+0].y_);
            p[2] = SIMD_SET_F4(points[i+3].z_, points[i+2].z_, points[i+1].z_, points[i+0].z_);

            float4 d2 = distanceSqr(o, d, p);
            int4 mask = SIMD_CAST_F4_I4(SIMD_CMPLT_F4(maxx4, d2));
            maxx4 = SIMD_MAX_F4(maxx4, d2);
            maxi4 = SIMD_OR_I4(SIMD_AND_I4(mask, index4), SIMD_ANDNOT_I4(mask, maxi4));
            index4 = SIMD_ADD_I4(index4, four);
        }
        OBB_ALIGN16 f32 dmaxx[Dim];
        OBB_ALIGN16 u32 dmaxi[Dim];
        SIMD_STORE_F4(dmaxx, maxx4);
        SIMD_STORE_I4(dmaxi, maxi4);

        f32 maxDistance = dmaxx[0];
        u32 maxi = dmaxi[0];
        for(u32 i = 1; i < Dim; ++i) {
            if(maxDistance < dmaxx[i]) {
                maxDistance = dmaxx[i];
                maxi = dmaxi[i];
            }
        }
        for(u32 i = size4; i < size; ++i) {
            f32 distance = distanceSqr(origin, direction, points[i]);
            if(maxDistance < distance) {
                maxDistance = distance;
                maxi = i;
            }
        }
        return std::make_tuple(maxDistance, maxi);
#endif
    }

    f32 findBestAxis(Vector3& b0, Vector3& b1, Vector3& b2, f32 minArea, const Vector3& normal, const Vector3& e0, const Vector3& e1, const Vector3& e2, const SoA8& soa)
    {
        Vector3 minx, maxx;
        Vector3 extent;
        extent.y_ = findMinMax(minx.y_, maxx.y_, normal, soa);

        f32 area = minArea;

        // edge0
        Vector3 axis0 = cross(e0, normal);
        extent.x_ = findMinMax(minx.x_, maxx.x_, e0, soa);
        extent.z_ = findMinMax(minx.z_, maxx.z_, axis0, soa);
        f32 area0 = halfBoxArea(extent);
        if(area0 < area) {
            area = area0;
            b0 = e0;
            b2 = axis0;
        }

        // edge1
        Vector3 axis1 = cross(e1, normal);
        extent.x_ = findMinMax(minx.x_, maxx.x_, e1, soa);
        extent.z_ = findMinMax(minx.z_, maxx.z_, axis1, soa);
        f32 area1 = halfBoxArea(extent);
        if(area1 < area) {
            area = area1;
            b0 = e1;
            b2 = axis1;
        }

        // edge0
        Vector3 axis2 = cross(e2, normal);
        extent.x_ = findMinMax(minx.x_, maxx.x_, e2, soa);
        extent.z_ = findMinMax(minx.z_, maxx.z_, axis2, soa);
        f32 area2 = halfBoxArea(extent);
        if(area2 < area) {
            area = area2;
            b0 = e2;
            b2 = axis2;
        }
        if(area < minArea) {
            b1 = normal;
        }
        return area;
    }

    std::tuple<u32, u32> findFurthestMinMaxOnPlain(const Vector3& origin, const Vector3& normal, const SoA8& soa)
    {
        f32 d = dot(origin, normal);
        float4 maxDistance = SIMD_SET1_F4(Epsilon);
        float4 minDistance = SIMD_SET1_F4(-Epsilon);
        int4 maxi = SIMD_SET1_I4(-1);
        int4 mini = SIMD_SET1_I4(-1);

        float4 nx = SIMD_SET1_F4(normal.x_);
        float4 ny = SIMD_SET1_F4(normal.y_);
        float4 nz = SIMD_SET1_F4(normal.z_);
        float4 d4 = SIMD_SET1_F4(d);
        int4 index = SIMD_SET_I4(3, 2, 1, 0);
        int4 four = SIMD_SET1_I4(4);
        for(u32 i = 0; i < InitialPoints; i += 4) {
            float4 px = SIMD_LOAD_F4(soa.x_ + i);
            float4 py = SIMD_LOAD_F4(soa.y_ + i);
            float4 pz = SIMD_LOAD_F4(soa.z_ + i);
            float4 mx = SIMD_MUL_F4(px, nx);
            float4 my = SIMD_MUL_F4(py, ny);
            float4 mz = SIMD_MUL_F4(pz, nz);
            float4 distance = SIMD_SUB_F4(SIMD_ADD_F4(mx, SIMD_ADD_F4(my, mz)), d4);
            float4 mask0 = SIMD_CMPLT_F4(maxDistance, distance);
            float4 mask1 = SIMD_CMPLT_F4(distance, minDistance);

            maxDistance = SIMD_OR_F4(SIMD_AND_F4(mask0, distance), SIMD_ANDNOT_F4(mask0, maxDistance));
            minDistance = SIMD_OR_F4(SIMD_AND_F4(mask1, distance), SIMD_ANDNOT_F4(mask1, minDistance));

            int4 maski0 = SIMD_CAST_F4_I4(mask0);
            int4 maski1 = SIMD_CAST_F4_I4(mask1);
            maxi = SIMD_OR_I4(SIMD_AND_I4(maski0, index), SIMD_ANDNOT_I4(maski0, maxi));
            mini = SIMD_OR_I4(SIMD_AND_I4(maski1, index), SIMD_ANDNOT_I4(maski1, mini));
            index = SIMD_ADD_I4(index, four);
        }
        OBB_ALIGN16 f32 tminx[Dim];
        OBB_ALIGN16 f32 tmaxx[Dim];
        OBB_ALIGN16 u32 tmini[Dim];
        OBB_ALIGN16 u32 tmaxi[Dim];
        SIMD_STORE_F4(tminx, minDistance);
        SIMD_STORE_F4(tmaxx, maxDistance);
        SIMD_STORE_I4(tmini, mini);
        SIMD_STORE_I4(tmaxi, maxi);
        for(u32 i = 1; i < Dim; ++i) {
            if(tminx[i] < tminx[0]) {
                tminx[0] = tminx[i];
                tmini[0] = tmini[i];
            }
            if(tmaxx[0] < tmaxx[i]) {
                tmaxx[0] = tmaxx[i];
                tmaxi[0] = tmaxi[i];
            }
        }
        return std::make_tuple(tmini[0], tmaxi[0]);
    }

    void axisAlignedOBB(OBB& obb, const Vector3& minx, const Vector3& maxx)
    {
        obb.center_ = (minx + maxx) * 0.5f;
        obb.axis0_ = {1.0f, 0.0f, 0.0f};
        obb.axis1_ = {0.0f, 1.0f, 0.0f};
        obb.axis2_ = {0.0f, 0.0f, 1.0f};
        obb.half_ = (maxx - minx) * 0.5f;
    }

    void lineAlignedOBB(OBB& obb, const Vector3& b0, u32 size, const Vector3* points)
    {
        Vector3 b1, b2;
        orthonormalBasis(b1, b2, b0);

        Vector3 minx, maxx;
        findMinMax(minx.x_, maxx.x_, b0, size, points);
        findMinMax(minx.y_, maxx.y_, b1, size, points);
        findMinMax(minx.z_, maxx.z_, b2, size, points);

        obb.axis0_ = b0;
        obb.axis1_ = b1;
        obb.axis2_ = b2;
        obb.half_ = (maxx - minx) * 0.5f;
        Vector3 center = 0.5f * (minx + maxx);
        obb.center_ = center.x_ * b0;
        obb.center_ += center.y_ * b1;
        obb.center_ += center.z_ * b2;
    }

    Vector3 getNormal(const SoA8& soa, u32 index, const Vector3& p)
    {
        Vector3 n;
        n.x_ = soa.x_[index] - p.x_;
        n.y_ = soa.y_[index] - p.y_;
        n.z_ = soa.z_[index] - p.z_;
        return normalize(n);
    }
} // namespace

void DiTO_simd(OBB& obb, u32 size, const Vector3* points)
{
    if(size<=0){
        obb = {};
        return;
    }

    OBB_ALIGN16 SoA8 minmax;
    auto [maxDistance, i0, i1] = findPoints(minmax, size, points);
    //for(u32 i = 0; i < InitialPoints; ++i) {
    //    printf("[%d] %f %f %f\n", i, minmax.x_[i], minmax.y_[i], minmax.z_[i]);
    //}
    //printf("%d %d\n", i0, i1);
    if(maxDistance < Epsilon) {
        Vector3 point0 = {minmax.x_[0], minmax.y_[2], minmax.z_[4]};
        Vector3 point1 = {minmax.x_[1], minmax.y_[3], minmax.z_[5]};
        axisAlignedOBB(obb, point0, point1);
        return;
    }
    Vector3 e0 = normalize(points[i1] - points[i0]);
    auto [maxLineDistance, i2] = findFurthestPoint(points[i0], e0, size, points);
    //printf("max:%f,%d\n", maxLineDistance, i2);
    if(maxLineDistance < Epsilon) {
        lineAlignedOBB(obb, e0, size, points);
        return;
    }
    OBB_ALIGN16 SoA8 candidates;
    for(u32 i = 0; i < InitialPoints; i += 4) {
        SIMD_STORE_F4(&candidates.x_[i], SIMD_LOAD_F4(&minmax.x_[i]));
        SIMD_STORE_F4(&candidates.y_[i], SIMD_LOAD_F4(&minmax.y_[i]));
        SIMD_STORE_F4(&candidates.z_[i], SIMD_LOAD_F4(&minmax.z_[i]));
    }
    if(size < InitialPoints) {
        for(u32 i = 0; i < size; ++i) {
            candidates.x_[i] = points[i].x_;
            candidates.y_[i] = points[i].y_;
            candidates.z_[i] = points[i].z_;
        }
        for(u32 i = size; i < InitialPoints; ++i) {
            candidates.x_[i] = points[size - 1].x_;
            candidates.y_[i] = points[size - 1].y_;
            candidates.z_[i] = points[size - 1].z_;
        }
    }
    const Vector3& p0 = points[i0];
    const Vector3& p1 = points[i1];
    const Vector3& p2 = points[i2];

    f32 minArea, aabbArea;
    Vector3 b0 = {1.0f, 0.0f, 0.0f};
    Vector3 b1 = {0.0f, 1.0f, 0.0f};
    Vector3 b2 = {0.0f, 0.0f, 1.0f};
    {
        Vector3 aabbExtent = {minmax.x_[1] - minmax.x_[0], minmax.y_[3] - minmax.y_[2], minmax.z_[5] - minmax.z_[4]};
        minArea = aabbArea = halfBoxArea(aabbExtent);
    }

    Vector3 e1 = normalize(p1 - p2);
    Vector3 e2 = normalize(p2 - p0);
    Vector3 normal = normalize(cross(e1, e0));
    { // Find from the triangle
        minArea = findBestAxis(b0, b1, b2, minArea, normal, e0, e1, e2, candidates);
    }
    { // Find from the two tetrahedron
        auto [mini, maxi] = findFurthestMinMaxOnPlain(p0, normal, candidates);
        if(Invalid != mini) {
            Vector3 t0 = getNormal(candidates, mini, p0);
            Vector3 t1 = getNormal(candidates, mini, p1);
            Vector3 t2 = getNormal(candidates, mini, p2);
            Vector3 n0 = normalize(cross(t1, e0));
            minArea = findBestAxis(b0, b1, b2, minArea, n0, e0, t1, t0, candidates);
            Vector3 n1 = normalize(cross(t2, e1));
            minArea = findBestAxis(b0, b1, b2, minArea, n1, e1, t2, t1, candidates);
            Vector3 n2 = normalize(cross(t0, e2));
            minArea = findBestAxis(b0, b1, b2, minArea, n2, e2, t0, t2, candidates);
        }
        if(Invalid != maxi) {
            Vector3 t0 = getNormal(candidates, maxi, p0);
            Vector3 t1 = getNormal(candidates, maxi, p1);
            Vector3 t2 = getNormal(candidates, maxi, p2);
            Vector3 n0 = normalize(cross(t1, e0));
            minArea = findBestAxis(b0, b1, b2, minArea, n0, e0, t1, t0, candidates);
            Vector3 n1 = normalize(cross(t2, e1));
            minArea = findBestAxis(b0, b1, b2, minArea, n1, e1, t2, t1, candidates);
            Vector3 n2 = normalize(cross(t0, e2));
            minArea = findBestAxis(b0, b1, b2, minArea, n2, e2, t0, t2, candidates);
        }
    }
    {
        Vector3 minx, maxx;
        findMinMax(minx.x_, maxx.x_, b0, size, points);
        findMinMax(minx.y_, maxx.y_, b1, size, points);
        findMinMax(minx.z_, maxx.z_, b2, size, points);
        //printf("(%f %f %f) - (%f %f %f)\n", minx.x_, minx.y_, minx.z_, maxx.x_, maxx.y_, maxx.z_);
        Vector3 extent = maxx - minx;
        minArea = halfBoxArea(extent);

        if(minArea < aabbArea) {
            obb.axis0_ = b0;
            obb.axis1_ = b1;
            obb.axis2_ = b2;
            obb.half_ = extent * 0.5f;
            Vector3 t = (minx + maxx) * 0.5f;
            obb.center_ = t.x_ * b0;
            obb.center_ += t.y_ * b1;
            obb.center_ += t.z_ * b2;
        } else {
            Vector3 point0 = {minmax.x_[0], minmax.y_[2], minmax.z_[4]};
            Vector3 point1 = {minmax.x_[1], minmax.y_[3], minmax.z_[5]};
            axisAlignedOBB(obb, point0, point1);
        }
    }
}

} // namespace obb
