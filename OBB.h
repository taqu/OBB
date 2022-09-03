#ifndef INC_OBB_H_
#define INC_OBB_H_
/**
*/
#include <cstdint>

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

//---------------------------------------------
struct OBB
{
    Vector3 center_;
    Vector3 axis0_;
    Vector3 axis1_;
    Vector3 axis2_;
    Vector3 half_;
};

//---------------------------------------------
/**
* @brief Calculate OBB by PCA
* @param [out] obb ... OBB
* @param [in] size ... number of points
* @param [in] points ... target points
 */
void PCA(OBB& obb, u32 size, const Vector3* points);

/**
* @brief Calculate OBB by 'Tight-Fitting Oriented Bounding Boxes'
* @param [out] obb ... OBB
* @param [in] size ... number of points
* @param [in] points ... target points
*/
void DiTO(OBB& obb, u32 size, const Vector3* points);


//--- Validation
//---------------------------------------------
/**
 * @brief Get 8 points from a OBB
 * @param [out] indices ... indices for 12 triangles
 * @param [out] points ... 8 points which compose a OBB
 * @param [in] obb ... target OBB
 */
void getPoints(u32 indices[36], Vector3 points[8], const OBB& obb);

struct Validation
{
    f32 maxDistance_; //!< the maximum distance of excluded points
    u32 countOuter_; //!< the number of excluded points
    f32 area_; //!< the area size of OBB surfaces
};

/**
 * @brief Validate quality of a generated OBB
 * @param [in] obb ... generated OBB
 * @param [in] size ... number of points
 * @param [in] points ... target points
 * @param [in] name ... filename of output
 */
Validation validate(const OBB& obb, u32 size, const Vector3* points, const char* name);
}
#endif //INC_OBB_H_
