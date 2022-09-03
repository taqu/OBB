/**
 */
#include "OBB.h"

#ifdef _DEBUG
#    include <stdio.h>
#endif

#include <cassert>
#include <limits>
#include <random>

namespace obb
{
namespace
{
    f32 absolute(f32 x)
    {
        return fabsf(x);
    }

    f32 maximum(f32 x0, f32 x1)
    {
        return x0 < x1 ? x1 : x0;
    }

    f32 minimum(f32 x0, f32 x1)
    {
        return x0 < x1 ? x0 : x1;
    }

    bool sameSign(f32 x0, f32 x1)
    {
        return (zero < x0) == (zero < x1);
    }
} // namespace

//---------------------------------------------
Vector3 Vector3::operator-() const
{
    return {-x_, -y_, -z_};
}

f32 Vector3::lengthSqr() const
{
    return x_ * x_ + y_ * y_ + z_ * z_;
}

Vector3& Vector3::operator+=(const Vector3& x)
{
    x_ += x.x_;
    y_ += x.y_;
    z_ += x.z_;
    return *this;
}

Vector3& Vector3::operator*=(f32 x)
{
    x_ *= x;
    y_ *= x;
    z_ *= x;
    return *this;
}

Vector3 operator+(const Vector3& x0, const Vector3& x1)
{
    return {x0.x_ + x1.x_, x0.y_ + x1.y_, x0.z_ + x1.z_};
}

Vector3 operator-(const Vector3& x0, const Vector3& x1)
{
    return {x0.x_ - x1.x_, x0.y_ - x1.y_, x0.z_ - x1.z_};
}

Vector3 operator*(f32 x0, const Vector3& x1)
{
    return {x0 * x1.x_, x0 * x1.y_, x0 * x1.z_};
}

Vector3 operator*(const Vector3& x0, f32 x1)
{
    return {x0.x_ * x1, x0.y_ * x1, x0.z_ * x1};
}

Vector3 minimum(const Vector3& x0, const Vector3& x1)
{
    return {
        minimum(x0.x_, x1.x_),
        minimum(x0.y_, x1.y_),
        minimum(x0.z_, x1.z_)};
}

Vector3 maximum(const Vector3& x0, const Vector3& x1)
{
    return {
        maximum(x0.x_, x1.x_),
        maximum(x0.y_, x1.y_),
        maximum(x0.z_, x1.z_)};
}

f32 dot(const Vector3& x0, const Vector3& x1)
{
    return x0.x_ * x1.x_ + x0.y_ * x1.y_ + x0.z_ * x1.z_;
}

Vector3 cross(const Vector3& x0, const Vector3& x1)
{
    f32 x = x0.y_ * x1.z_ - x0.z_ * x1.y_;
    f32 y = x0.z_ * x1.x_ - x0.x_ * x1.z_;
    f32 z = x0.x_ * x1.y_ - x0.y_ * x1.x_;
    return {x, y, z};
}

Vector3 normalize(const Vector3& x)
{
    f32 inv = 1.0f / ::sqrtf(x.lengthSqr());
    return {x.x_ * inv, x.y_ * inv, x.z_ * inv};
}

Vector3 normalizeSafe(const Vector3& x)
{
    f32 l = ::sqrtf(x.lengthSqr());
    if(l < Epsilon) {
        return {0.0f, 0.0f, 0.0f};
    }
    f32 inv = 1.0f / l;
    return {x.x_ * inv, x.y_ * inv, x.z_ * inv};
}

f32 distanceSqr(const Vector3& x0, const Vector3& x1)
{
    Vector3 d = x1 - x0;
    return dot(d, d);
}

void orthonormalBasis(Vector3& binormal0, Vector3& binormal1, const Vector3& normal)
{
    if(normal.z_ < -0.999f) {
        binormal0 = {0.0f, -1.0f, 0.0f};
        binormal1 = {-1.0f, 0.0f, 0.0f};
        return;
    }

    const f32 a = 1.0f / (1.0f + normal.z_);
    const f32 b = -normal.x_ * normal.y_ * a;
    binormal0 = {1.0f - normal.x_ * normal.x_ * a, b, -normal.x_};
    binormal1 = {b, 1.0f - normal.y_ * normal.y_ * a, -normal.y_};
}

namespace
{
    /**
     * @brief Kahan's summation
     */
    Vector3 kahan(u32 size, const Vector3* points)
    {
        Vector3 sum = {};
        Vector3 error = {};
        for(u32 i = 0; i < size; ++i) {
            Vector3 y = points[i] - error;
            Vector3 t = sum + y;
            error = (t - sum) - y;
            sum = t;
        }
        return sum;
    }

    /**
     * @brief Kahan's summation
     */
    void kahan(f32* OBB_RESTRICT sum, f32* OBB_RESTRICT error, f32 x)
    {
        f32 y = x - *error;
        f32 t = *sum + y;
        *error = (t - *sum) - y;
        *sum = t;
    }

    /**
     * @brief Kahan's summation
     */
    void kahan(f32 covariance[9], u32 size, const Vector3* points, const Vector3& average)
    {
        f32 errors[6] = {};
        for(u32 i = 0; i < size; ++i) {
            f32 dx = points[i].x_ - average.x_;
            f32 dy = points[i].y_ - average.y_;
            f32 dz = points[i].z_ - average.z_;
            kahan(&covariance[0], &errors[0], dx * dx);
            kahan(&covariance[1], &errors[1], dx * dy);
            kahan(&covariance[2], &errors[2], dx * dz);
            kahan(&covariance[4], &errors[3], dy * dy);
            kahan(&covariance[5], &errors[4], dy * dz);
            kahan(&covariance[8], &errors[5], dz * dz);
        }
    }

    void rotate(f32 m[9], u32 i, u32 j, u32 k, u32 l, f32 cs, f32 sn)
    {
        f32 t0 = m[i * 3 + j];
        f32 t1 = m[k * 3 + l];
        m[i * 3 + j] = t0 * cs - t1 * sn;
        m[k * 3 + l] = t0 * sn + t1 * cs;
    }

    bool jacobi(f32 M[9], f32 N[9])
    {
        static constexpr u32 Size = 3;
        static constexpr u32 MaxIteration = 100;
        ::memset(N, 0, sizeof(f32) * 9);
        f32 cs;
        f32 sn = 0.0f;
        f32 offdiag = 0.0f;
        for(u32 i = 0; i < Size; ++i) {
            N[i * Size + i] = 1.0f;
            u32 t = i * Size + i;
            sn += M[t] * M[t];
            for(u32 j = i + 1; j < Size; ++j) {
                t = i * Size + j;
                offdiag += M[t] * M[t];
            }
        } // for(u32 i
        f32 tolerance = Epsilon * Epsilon * (sn * 0.5f + offdiag);
        u32 iteration = 0;
        for(; iteration < MaxIteration; ++iteration) {
            offdiag = 0.0f;
            for(u32 i = 0; i < Size - 1; ++i) {
                for(u32 j = i + 1; j < Size; ++j) {
                    u32 t = i * 3 + j;
                    offdiag += M[t] * M[t];
                }
            }
            if(offdiag < tolerance) {
                break;
            }
            for(u32 i = 0; i < Size - 1; ++i) {
                for(u32 j = i + 1; j < Size; ++j) {
                    if(::fabsf(M[i * Size + j]) < AlmostZero) {
                        continue;
                    }
                    f32 t = (M[j * Size + j] - M[i * Size + i]) / (2.0f * M[i * Size + j]);
                    if(0.0f <= t) {
                        t = 1.0f / (t + ::sqrtf(t * t + 1.0f));
                    } else {
                        t = 1.0f / (t - ::sqrtf(t * t + 1.0f));
                    }
                    cs = 1.0f / ::sqrtf(t * t + 1.0f);
                    sn = t * cs;
                    t *= M[i * Size + j];
                    M[i * Size + i] -= t;
                    M[j * Size + j] += t;
                    M[i * Size + j] = 0.0f;
                    for(u32 k = 0; k < i; ++k) {
                        rotate(M, k, i, k, j, cs, sn);
                    }
                    for(u32 k = i + 1; k < j; ++k) {
                        rotate(M, i, k, k, j, cs, sn);
                    }
                    for(u32 k = j + 1; k < Size; ++k) {
                        rotate(M, i, k, j, k, cs, sn);
                    }
                    for(u32 k = 0; k < Size; ++k) {
                        rotate(N, i, k, j, k, cs, sn);
                    }
                } // for(u32 j
            }     // for(u32 i
        }         // for(u32 iteration
        return (iteration < MaxIteration);
    }

    //
    static constexpr f32 Sqrt3 = static_cast<f32>(0.57735026918962576450914878050196);
    static const Vector3 N07[7] =
    {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {Sqrt3, Sqrt3, Sqrt3},
        {Sqrt3, Sqrt3, -Sqrt3},
        {Sqrt3, -Sqrt3, Sqrt3},
        {Sqrt3, -Sqrt3, -Sqrt3},
    };

    f32 halfBoxArea(const Vector3& extent)
    {
        return extent.x_ * extent.y_ + extent.y_ * extent.z_ + extent.z_ * extent.x_;
    }

    f32 findMinMax(f32& minx, f32& maxx, const Vector3& normal, u32 size, const Vector3* points)
    {
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
    }

    /**
     * @brief Find the min and max points along normals
     * @param [out] results ... min and max points for each normals
     * @param [in] size ... number of points
     * @param [in] points ... target points
     * @param [in] numNormals ... number of normals
     * @param [in] normals ... finding along these normals
     * @return a tuple of a maximum distance, a pair of indices
     */
    std::tuple<f32, u32, u32> findPoints(Vector3* OBB_RESTRICT results, u32 size, const Vector3* OBB_RESTRICT points, u32 numNormals, const Vector3* OBB_RESTRICT normals)
    {
        u32 count = 0;
        f32 maxDistance = 0.0f;
        u32 pair0;
        u32 pair1;
        for(u32 n = 0; n < numNormals; ++n) {
            f32 minx = std::numeric_limits<f32>::infinity();
            f32 maxx = -std::numeric_limits<f32>::infinity();
            u32 mini = Invalid;
            u32 maxi = Invalid;
            const Vector3& N = normals[n];
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
            results[count++] = points[mini];
            results[count++] = points[maxi];
        }
        return std::make_tuple(maxDistance, pair0, pair1);
    }

    f32 distanceSqr(const Vector3& origin, const Vector3& direction, const Vector3& point)
    {
        Vector3 diff = origin - point;
        f32 d = dot(diff, direction);
        Vector3 p = diff - d * direction;
        return p.lengthSqr();
    }

    std::tuple<f32, u32> findFurthestPoint(const Vector3& origin, const Vector3& direction, u32 size, const Vector3* points)
    {
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
    }

    f32 findBestAxis(Vector3& b0, Vector3& b1, Vector3& b2, f32 minArea, const Vector3& normal, const Vector3& e0, const Vector3& e1, const Vector3& e2, u32 size, const Vector3* points)
    {
        Vector3 minx, maxx;
        Vector3 extent;
        extent.y_ = findMinMax(minx.y_, maxx.y_, normal, size, points);

        f32 area = minArea;

        // edge0
        Vector3 axis0 = cross(e0, normal);
        extent.x_ = findMinMax(minx.x_, maxx.x_, e0, size, points);
        extent.z_ = findMinMax(minx.z_, maxx.z_, axis0, size, points);
        f32 area0 = halfBoxArea(extent);
        if(area0 < area) {
            area = area0;
            b0 = e0;
            b2 = axis0;
        }

        // edge1
        Vector3 axis1 = cross(e1, normal);
        extent.x_ = findMinMax(minx.x_, maxx.x_, e1, size, points);
        extent.z_ = findMinMax(minx.z_, maxx.z_, axis1, size, points);
        f32 area1 = halfBoxArea(extent);
        if(area1 < area) {
            area = area1;
            b0 = e1;
            b2 = axis1;
        }

        // edge0
        Vector3 axis2 = cross(e2, normal);
        extent.x_ = findMinMax(minx.x_, maxx.x_, e2, size, points);
        extent.z_ = findMinMax(minx.z_, maxx.z_, axis2, size, points);
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

    std::tuple<u32, u32> findFurthestMinMaxOnPlain(const Vector3& origin, const Vector3& normal, u32 size, const Vector3* points)
    {
        f32 d = dot(origin, normal);
        f32 maxDistance = Epsilon;
        f32 minDistance = -Epsilon;
        u32 maxi = Invalid;
        u32 mini = Invalid;
        for(u32 i = 0; i < size; ++i) {
            f32 distance = dot(points[i], normal) - d;
            if(maxDistance < distance) {
                maxDistance = distance;
                maxi = i;
            }
            if(distance < minDistance) {
                minDistance = distance;
                mini = i;
            }
        }
        return std::make_tuple(mini, maxi);
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
}
// namespace

void PCA(OBB& obb, u32 size, const Vector3* points)
{
    if(size<=0){
        obb = {};
        return;
    }
    Vector3 average = kahan(size, points);
    f32 invSize = 1.0f / size;
    average *= invSize;

    f32 covariance[9] = {};
    kahan(covariance, size, points, average);
    covariance[0] *= invSize;
    covariance[1] *= invSize;
    covariance[2] *= invSize;
    covariance[4] *= invSize;
    covariance[5] *= invSize;
    covariance[8] *= invSize;

    covariance[3] = covariance[1];
    covariance[6] = covariance[2];
    covariance[7] = covariance[5];
    f32 N[9];
    if(!jacobi(covariance, N)) {
        obb = {};
        return;
    }
    obb.axis0_ = normalize({N[0], N[1], N[2]});
    obb.axis1_ = normalize({N[3], N[4], N[5]});
    obb.axis2_ = normalize({N[6], N[7], N[8]});
    f32 d0 = dot(obb.axis0_, average);
    f32 d1 = dot(obb.axis1_, average);
    f32 d2 = dot(obb.axis2_, average);
    Vector3 minP = {};
    Vector3 maxP = {};
    for(u32 i = 0; i < size; ++i) {
        f32 dx = dot(obb.axis0_, points[i]) - d0;
        f32 dy = dot(obb.axis1_, points[i]) - d1;
        f32 dz = dot(obb.axis2_, points[i]) - d2;
        minP.x_ = minimum(dx, minP.x_);
        minP.y_ = minimum(dy, minP.y_);
        minP.z_ = minimum(dz, minP.z_);

        maxP.x_ = maximum(dx, maxP.x_);
        maxP.y_ = maximum(dy, maxP.y_);
        maxP.z_ = maximum(dz, maxP.z_);
    }
    obb.half_.x_ = (maxP.x_ - minP.x_) * 0.5f;
    obb.half_.y_ = (maxP.y_ - minP.y_) * 0.5f;
    obb.half_.z_ = (maxP.z_ - minP.z_) * 0.5f;
    Vector3 p0 = average + (minP.x_ * obb.axis0_ + maxP.x_ * obb.axis0_) * 0.5f;
    Vector3 p1 = average + (minP.y_ * obb.axis1_ + maxP.y_ * obb.axis1_) * 0.5f;
    Vector3 p2 = average + (minP.z_ * obb.axis2_ + maxP.z_ * obb.axis2_) * 0.5f;
    Vector3 d = (minP + maxP) * 0.5f;
    obb.center_ = average + obb.axis0_ * d.x_ + obb.axis1_ * d.y_ + obb.axis2_ * d.z_;
}

void DiTO(OBB& obb, u32 size, const Vector3* points)
{
    if(size<=0){
        obb = {};
        return;
    }
    static const u32 NumPoints = 14;
    Vector3 minmax[NumPoints];
    auto [maxDistance, i0, i1] = findPoints(minmax, size, points, NumPoints/2, N07);

    if(maxDistance < Epsilon) {
        Vector3 point0 = {minmax[0].x_, minmax[2].y_, minmax[4].z_};
        Vector3 point1 = {minmax[1].x_, minmax[3].y_, minmax[5].z_};
        axisAlignedOBB(obb, point0, point1);
        return;
    }
    Vector3 e0 = normalize(points[i1] - points[i0]);
    auto [maxLineDistance, i2] = findFurthestPoint(points[i0], e0, size, points);
    if(maxLineDistance < Epsilon) {
        lineAlignedOBB(obb, e0, size, points);
        return;
    }
    u32 numCandidates = NumPoints;
    const Vector3* candidates = minmax;
    if(size < NumPoints) {
        numCandidates = size;
        candidates = points;
    } else {
        numCandidates = NumPoints;
        candidates = minmax;
    }
    const Vector3& p0 = points[i0];
    const Vector3& p1 = points[i1];
    const Vector3& p2 = points[i2];

    f32 minArea, aabbArea;
    Vector3 b0 = {1.0f, 0.0f, 0.0f};
    Vector3 b1 = {0.0f, 1.0f, 0.0f};
    Vector3 b2 = {0.0f, 0.0f, 1.0f};
    {
        Vector3 aabbExtent = {minmax[1].x_ - minmax[0].x_, minmax[3].y_ - minmax[2].y_, minmax[5].z_ - minmax[4].z_};
        minArea = aabbArea = halfBoxArea(aabbExtent);
    }

    Vector3 e1 = normalize(p1 - p2);
    Vector3 e2 = normalize(p2 - p0);
    Vector3 normal = normalize(cross(e1, e0));
    { // Find from the triangle
        minArea = findBestAxis(b0, b1, b2, minArea, normal, e0, e1, e2, numCandidates, candidates);
    }
    { // Find from the two tetrahedron
        auto [mini, maxi] = findFurthestMinMaxOnPlain(p0, normal, numCandidates, candidates);
        if(Invalid != mini) {
            Vector3 t0 = normalize(candidates[mini] - p0);
            Vector3 t1 = normalize(candidates[mini] - p1);
            Vector3 t2 = normalize(candidates[mini] - p2);
            Vector3 n0 = normalize(cross(t1, e0));
            minArea = findBestAxis(b0, b1, b2, minArea, n0, e0, t1, t0, numCandidates, candidates);
            Vector3 n1 = normalize(cross(t2, e1));
            minArea = findBestAxis(b0, b1, b2, minArea, n1, e1, t2, t1, numCandidates, candidates);
            Vector3 n2 = normalize(cross(t0, e2));
            minArea = findBestAxis(b0, b1, b2, minArea, n2, e2, t0, t2, numCandidates, candidates);
        }
        if(Invalid != maxi) {
            Vector3 t0 = normalize(candidates[maxi] - p0);
            Vector3 t1 = normalize(candidates[maxi] - p1);
            Vector3 t2 = normalize(candidates[maxi] - p2);
            Vector3 n0 = normalize(cross(t1, e0));
            minArea = findBestAxis(b0, b1, b2, minArea, n0, e0, t1, t0, numCandidates, candidates);
            Vector3 n1 = normalize(cross(t2, e1));
            minArea = findBestAxis(b0, b1, b2, minArea, n1, e1, t2, t1, numCandidates, candidates);
            Vector3 n2 = normalize(cross(t0, e2));
            minArea = findBestAxis(b0, b1, b2, minArea, n2, e2, t0, t2, numCandidates, candidates);
        }
    }
    {
        Vector3 minx, maxx;
        findMinMax(minx.x_, maxx.x_, b0, size, points);
        findMinMax(minx.y_, maxx.y_, b1, size, points);
        findMinMax(minx.z_, maxx.z_, b2, size, points);
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
            Vector3 point0 = {minmax[0].x_, minmax[2].y_, minmax[4].z_};
            Vector3 point1 = {minmax[1].x_, minmax[3].y_, minmax[5].z_};
            axisAlignedOBB(obb, point0, point1);
        }
    }
}

void getPoints(u32 indices[36], Vector3 points[8], const OBB& obb)
{
    //Get 8 corner points
    points[0] = -(obb.axis0_ * obb.half_.x_) + (obb.axis1_ * obb.half_.y_) + (obb.axis2_ * obb.half_.z_) + obb.center_;
    points[1] = (obb.axis0_ * obb.half_.x_) + (obb.axis1_ * obb.half_.y_) + (obb.axis2_ * obb.half_.z_) + obb.center_;
    points[2] = (obb.axis0_ * obb.half_.x_) - (obb.axis1_ * obb.half_.y_) + (obb.axis2_ * obb.half_.z_) + obb.center_;
    points[3] = -(obb.axis0_ * obb.half_.x_) - (obb.axis1_ * obb.half_.y_) + (obb.axis2_ * obb.half_.z_) + obb.center_;

    points[4] = -(obb.axis0_ * obb.half_.x_) + (obb.axis1_ * obb.half_.y_) - (obb.axis2_ * obb.half_.z_) + obb.center_;
    points[5] = (obb.axis0_ * obb.half_.x_) + (obb.axis1_ * obb.half_.y_) - (obb.axis2_ * obb.half_.z_) + obb.center_;
    points[6] = (obb.axis0_ * obb.half_.x_) - (obb.axis1_ * obb.half_.y_) - (obb.axis2_ * obb.half_.z_) + obb.center_;
    points[7] = -(obb.axis0_ * obb.half_.x_) - (obb.axis1_ * obb.half_.y_) - (obb.axis2_ * obb.half_.z_) + obb.center_;

    // Make triangle indices
    ::memset(indices, 0, sizeof(s32) * 36);
    // 0
    indices[0] = 1;
    indices[1] = 0;
    indices[2] = 2;

    indices[3] = 2;
    indices[4] = 0;
    indices[5] = 3;

    // 1
    indices[6] = 0;
    indices[7] = 1;
    indices[8] = 4;

    indices[9] = 5;
    indices[10] = 4;
    indices[11] = 1;

    // 2
    indices[12] = 1;
    indices[13] = 2;
    indices[14] = 5;

    indices[15] = 6;
    indices[16] = 5;
    indices[17] = 2;

    // 3
    indices[18] = 2;
    indices[19] = 3;
    indices[20] = 6;

    indices[21] = 7;
    indices[22] = 6;
    indices[23] = 3;

    // 4
    indices[24] = 3;
    indices[25] = 0;
    indices[26] = 7;

    indices[27] = 4;
    indices[28] = 7;
    indices[29] = 0;

    // 5
    indices[30] = 4;
    indices[31] = 5;
    indices[32] = 6;

    indices[33] = 4;
    indices[34] = 6;
    indices[35] = 7;
}

Validation validate(const OBB& obb, u32 size, const Vector3* points, const char* name)
{
    u32 triangles = 12;
    FILE* file = fopen(name, "wb");
    if(nullptr == file) {
        return {0.0f, 0};
    }
    fprintf(file, "ply\nformat ascii 1.0\n");
    fprintf(file, "element vertex %d\n", size + 8);
    fprintf(file, "property float x\nproperty float y\nproperty float z\n");
    fprintf(file, "element face %d\n", triangles);
    fprintf(file, "property list uchar int vertex_index\n");
    fprintf(file, "end_header\n");
    for(u32 i = 0; i < size; ++i) {
        fprintf(file, "%f %f %f\n", points[i].x_, points[i].y_, points[i].z_);
    }
    u32 indices[36];
    Vector3 corners[8];
    getPoints(indices, corners, obb);
    for(u32 i = 0; i < 8; ++i) {
        fprintf(file, "%f %f %f\n", corners[i].x_, corners[i].y_, corners[i].z_);
    }

    for(u32 i = 0; i < triangles; ++i) {
        u32 n = i * 3;
        fprintf(file, "3 %d %d %d\n", indices[n + 0] + size, indices[n + 1] + size, indices[n + 2] + size);
    }
    fclose(file);

    u32 count = 0;
    f32 d0 = dot(obb.axis0_, obb.center_);
    f32 d1 = dot(obb.axis1_, obb.center_);
    f32 d2 = dot(obb.axis2_, obb.center_);
    f32 maxDistance = 0.0f;
    for(u32 i = 0; i < size; ++i) {
        f32 h0 = ::fabsf(dot(points[i], obb.axis0_) - d0);
        f32 h1 = ::fabsf(dot(points[i], obb.axis1_) - d1);
        f32 h2 = ::fabsf(dot(points[i], obb.axis2_) - d2);
        if(obb.half_.x_ < h0 || obb.half_.y_ < h1 || obb.half_.z_ < h2) {
            maxDistance = maximum(maxDistance, h0 - obb.half_.x_);
            maxDistance = maximum(maxDistance, h1 - obb.half_.y_);
            maxDistance = maximum(maxDistance, h2 - obb.half_.z_);
            ++count;
        }
    }
    f32 area = obb.half_.x_ * obb.half_.y_ + obb.half_.y_ * obb.half_.z_ + obb.half_.z_ * obb.half_.x_;
    return {maxDistance, count, area};
}
} // namespace obb
