#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <random>
#include <tuple>

#include "OBB.h"
#include "OBB_simd.h"

#define ALIGNED_ALLOC(size, align) _aligned_malloc((size), (align))
#define ALIGNED_FREE(ptr) _aligned_free(ptr)

using namespace obb;

static constexpr u64 Increment = 0x14057B7EF767814FULL;
static constexpr u64 Multiplier = 0x5851F42D4C957F2DULL;
inline f32 toF32_1(u32 x)
{
    static const u32 m0 = 0x3F800000U;
    static const u32 m1 = 0x007FFFFFU;
    x = m0 | (x & m1);
    return (*(f32*)&x) - 1.000000000f;
}

u64 state_ = 0xCAFEF00DD15EA5E5ULL;

namespace
{
inline u32 rotr32(u32 x, u32 r)
{
    return (x >> r) | (x << ((~r + 1) & 31U));
}
} // namespace

u32 urand()
{
    u64 x = state_;
    u32 count = static_cast<u32>(x >> 59);
    state_ = x * Multiplier + Increment;
    x ^= x >> 18;
    return rotr32(static_cast<u32>(x >> 27), count);
}

f32 frand()
{
    return toF32_1(urand());
}

void srandom(u64 seed)
{
    state_ = Increment + seed;
    urand();
}

void randomBasis(Vector3& b0, Vector3& b1, Vector3& b2)
{
    for(;;) {
        b2.x_ = frand();
        b2.y_ = frand();
        b2.z_ = frand();
        f32 l = b2.lengthSqr();
        if(l < 1.0e-5f) {
            continue;
        }
        b2 = normalize(b2);
        break;
    }
    orthonormalBasis(b0, b1, b2);
}

Vector3 rotate(const Vector3& x, const Vector3& b0, const Vector3& b1, const Vector3& b2)
{
    return {dot(x, b0), dot(x, b1), dot(x, b2)};
}

std::tuple<std::chrono::nanoseconds, f32, u32, f32> procPCA(u32 numSamples, u32 count, f32 scale0, f32 scale1)
{
    Vector3* points = reinterpret_cast<Vector3*>(malloc(sizeof(Vector3) * numSamples));
    Vector3 b0, b1, b2;
    randomBasis(b0, b1, b2);
    u32 half = numSamples / 2;
    for(u32 i = 0; i < half; ++i) {
        points[i].x_ = frand() * scale0;
        points[i].y_ = frand() * scale0;
        points[i].z_ = frand() * scale1;
        points[i] = rotate(points[i], b0, b1, b2);
    }
    randomBasis(b0, b1, b2);
    for(u32 i = half; i < numSamples; ++i) {
        points[i].x_ = frand() * scale0;
        points[i].y_ = frand() * scale0;
        points[i].z_ = frand() * scale1;
        points[i] = rotate(points[i], b0, b1, b2);
    }
    OBB result;
    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    PCA(result, numSamples, points);
    std::chrono::duration duration = std::chrono::high_resolution_clock::now() - start;
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "./out/obb_pca_%04d_%04d.ply", numSamples, count);
    auto validation = validate(result, numSamples, points, buffer);
    free(points);
    return std::make_tuple(std::chrono::duration_cast<std::chrono::nanoseconds>(duration), validation.maxDistance_, validation.countOuter_, validation.area_);
}

std::tuple<std::chrono::nanoseconds, f32, u32, f32> procDiTO(u32 numSamples, u32 count, f32 scale0, f32 scale1)
{
    Vector3* points = reinterpret_cast<Vector3*>(malloc(sizeof(Vector3) * numSamples));
    Vector3 b0, b1, b2;
    randomBasis(b0, b1, b2);
    u32 half = numSamples / 2;
    for(u32 i = 0; i < half; ++i) {
        points[i].x_ = frand() * scale0;
        points[i].y_ = frand() * scale0;
        points[i].z_ = frand() * scale1;
        points[i] = rotate(points[i], b0, b1, b2);
    }
    randomBasis(b0, b1, b2);
    for(u32 i = half; i < numSamples; ++i) {
        points[i].x_ = frand() * scale0;
        points[i].y_ = frand() * scale0;
        points[i].z_ = frand() * scale1;
        points[i] = rotate(points[i], b0, b1, b2);
    }
    OBB result;
    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    DiTO(result, numSamples, points);
    std::chrono::duration duration = std::chrono::high_resolution_clock::now() - start;
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "./out/obb_dito_%04d_%04d.ply", numSamples, count);
    auto validation = validate(result, numSamples, points, buffer);
    free(points);
    return std::make_tuple(std::chrono::duration_cast<std::chrono::nanoseconds>(duration), validation.maxDistance_, validation.countOuter_, validation.area_);
}

std::tuple<std::chrono::nanoseconds, f32, u32, f32> procDiTO_simd(u32 numSamples, u32 count, f32 scale0, f32 scale1)
{
    Vector3* points = reinterpret_cast<Vector3*>(malloc(sizeof(Vector3) * numSamples));
    Vector3 b0, b1, b2;
    randomBasis(b0, b1, b2);
    u32 half = numSamples / 2;
    for(u32 i = 0; i < half; ++i) {
        points[i].x_ = frand() * scale0;
        points[i].y_ = frand() * scale0;
        points[i].z_ = frand() * scale1;
        points[i] = rotate(points[i], b0, b1, b2);
    }
    randomBasis(b0, b1, b2);
    for(u32 i = half; i < numSamples; ++i) {
        points[i].x_ = frand() * scale0;
        points[i].y_ = frand() * scale0;
        points[i].z_ = frand() * scale1;
        points[i] = rotate(points[i], b0, b1, b2);
    }
    OBB result;
    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    DiTO_simd(result, numSamples, points);
    std::chrono::duration duration = std::chrono::high_resolution_clock::now() - start;
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "./out/obb_dito_simd_%04d_%04d.ply", numSamples, count);
    auto validation = validate(result, numSamples, points, buffer);
    free(points);
    return std::make_tuple(std::chrono::duration_cast<std::chrono::nanoseconds>(duration), validation.maxDistance_, validation.countOuter_, validation.area_);
}

using func_type = std::function<std::tuple<std::chrono::nanoseconds, f32, u32, f32>(u32, u32, f32, f32)>;

void proc(u32 numSamples, s32 count, f32 scale0, f32 scale1, u32 seed, const char* name, func_type func)
{
    printf("---\nobb %s %d samples\n---\n", name, numSamples);
    std::chrono::nanoseconds maxDuration = {};
    std::chrono::nanoseconds totalDuration = {};
    f32 maxDistance = -std::numeric_limits<f32>::infinity();
    f64 totalDistance = 0.0;
    u32 maxCount = 0;
    u64 totalCount = 0;
    f32 minArea = std::numeric_limits<f32>::infinity();
    f32 maxArea = -std::numeric_limits<f32>::infinity();
    f64 totalArea = 0.0;
    srandom(seed);
    for(s32 i = 0; i < count; ++i) {
        auto statistics = func(numSamples, i, scale0, scale1);
        long long time = std::chrono::duration_cast<std::chrono::microseconds>(std::get<0>(statistics)).count();
        printf("[%d] %lld microseconds\n", i, time);
        if(maxDuration < std::get<0>(statistics)) {
            maxDuration = std::get<0>(statistics);
        }
        totalDuration += std::get<0>(statistics);
        if(maxDistance < std::get<1>(statistics)) {
            maxDistance = std::get<1>(statistics);
        }
        totalDistance += std::get<1>(statistics);
        if(maxCount < std::get<2>(statistics)) {
            maxCount = std::get<2>(statistics);
        }
        totalCount += std::get<2>(statistics);
        totalArea += std::get<3>(statistics);
        if(std::get<3>(statistics) < minArea) {
            minArea = std::get<3>(statistics);
        }
        if(maxArea < std::get<3>(statistics)) {
            maxArea = std::get<3>(statistics);
        }
    }

    char filename[128];
    snprintf(filename, 128, "statistics_%s.txt", name);
    FILE* file = fopen(filename, "wb");
    fprintf(file, "cout: %d, points: %d\n", count, numSamples);
    std::chrono::nanoseconds avgDuration = totalDuration / count;
    f64 avgDistance = totalDistance / count;
    u64 avgCount = totalCount / count;
    f64 avgArea = totalArea / count;

    long long avgTime = std::chrono::duration_cast<std::chrono::microseconds>(avgDuration).count();
    long long maxTime = std::chrono::duration_cast<std::chrono::microseconds>(maxDuration).count();

    fprintf(file, "[time] average: %lld microseconds, max: %lld microseconds\n", avgTime, maxTime);
    fprintf(file, "[distance] average: %f, max: %f\n", avgDistance, maxDistance);
    fprintf(file, "[count] average: %zu, max: %d\n", avgCount, maxCount);
    fprintf(file, "[area] average: %f, min: %f, max: %f\n", avgArea, minArea, maxArea);
    fclose(file);
}

int main(void)
{
    static constexpr s32 Samples = 60001;
    // u32 seed = 123456U;
    u32 seed = std::random_device()();
    static constexpr s32 Count = 1000;
    static constexpr f32 Scale0 = 1.0e3f;
    static constexpr f32 Scale1 = 1.0f;

#if 0
    for(s32 i=0; i<8; ++i){
        proc(i, 10, Scale0, Scale1, seed, "pca", procPCA);
    proc(i, 10, Scale0, Scale1, seed, "gito", procDiTO);
    proc(i, 10, Scale0, Scale1, seed, "gito_simd", procDiTO_simd);
    }
#else
    proc(Samples, Count, Scale0, Scale1, seed, "pca", procPCA);
    proc(Samples, Count, Scale0, Scale1, seed, "gito", procDiTO);
    proc(Samples, Count, Scale0, Scale1, seed, "gito_simd", procDiTO_simd);
#endif
    return 0;
}
