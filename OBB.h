#ifndef INC_OBB_H_
#define INC_OBB_H_
/**
*/
#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace obb
{
    using s32 = int32_t;

    using u8 = uint8_t;
    using u16 = uint16_t;
    using u32 = uint32_t;
    using u64 = uint64_t;

    using f32 = float;
    using f64 = double;

#if defined(_MSC_VER)
#    define OBB_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#    define OBB_RESTRICT __restrict__
#else
#    define OBB_RESTRICT
#endif

    static constexpr f32 zero = 0.0f;
    static constexpr f32 one = 1.0f;

    static constexpr f32 Epsilon = 1.0e-6f;
    static constexpr f32 AlmostZero = 1.0e-20f;
    static constexpr u32 Invalid = 0xFFFFFFFFU;

    enum class Result
    {
        Error,
        Point,
        Colinear,
        Coplanar,
        Convexhull,
    };

    //---------------------------------------------
    struct Vector3
    {
        f32 x_;
        f32 y_;
        f32 z_;

        Vector3 operator-() const;

        f32 operator[](u32 index) const
        {
            return reinterpret_cast<const f32*>(this)[index];
        }
        f32 lengthSqr() const;
        Vector3& operator+=(const Vector3& x);
        Vector3& operator*=(f32 x);
    };

    Vector3 operator+(const Vector3& x0, const Vector3& x1);
    Vector3 operator-(const Vector3& x0, const Vector3& x1);
    Vector3 operator*(f32 x0, const Vector3& x1);
    Vector3 operator*(const Vector3& x0, f32 x1);

    Vector3 minimum(const Vector3& x0, const Vector3& x1);
    Vector3 maximum(const Vector3& x0, const Vector3& x1);

    f32 dot(const Vector3& x0, const Vector3& x1);
    Vector3 cross(const Vector3& x0, const Vector3& x1);
    Vector3 normalize(const Vector3& x);
    Vector3 normalizeSafe(const Vector3& x);
    f32 distanceSqr(const Vector3& x0, const Vector3& x1);
    void orthonormalBasis(Vector3& binormal0, Vector3& binormal1, const Vector3& normal);

    struct OBB
    {
        Vector3 center_;
        Vector3 axis0_;
        Vector3 axis1_;
        Vector3 axis2_;
        Vector3 half_;
    };

    void PCA(OBB& obb, u32 size, const Vector3* points);
    void DiTO(OBB& obb, u32 size, const Vector3* points);
    void getPoints(u32 indices[36], Vector3 points[8], const OBB& obb);

    //---------------------------------------------
    struct Validation
    {
        f32 maxDistance_;
        u32 countOuter_;
        f32 area_;
    };
    Validation validate(const OBB& obb, u32 size, const Vector3* points, const char* name);
}
#endif //INC_OBB_H_

