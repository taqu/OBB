#ifndef INC_OBB_SIMD_H_
#define INC_OBB_SIMD_H_
/**
*/
#include "OBB.h"

namespace obb
{
/**
* @brief Calculate OBB by 'Tight-Fitting Oriented Bounding Boxes'
* @param [out] obb ... OBB
* @param [in] size ... number of points
* @param [in] points ... target points
*/
void DiTO_simd(OBB& obb, u32 size, const Vector3* points);
}
#endif //INC_OBB_SIMD_H_
