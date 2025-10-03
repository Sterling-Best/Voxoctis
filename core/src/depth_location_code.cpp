// src/depth_location_code.cpp
#include "depth_location_code.hpp"

namespace Voxoctis {

// Explicit instantiations for common codeType/posType combinations
template class DepthLocationCode<uint8_t,  uint8_t>;
template class DepthLocationCode<uint8_t,  uint16_t>;
template class DepthLocationCode<uint8_t,  uint32_t>;
template class DepthLocationCode<uint8_t,  uint64_t>;

template class DepthLocationCode<uint16_t, uint8_t>;
template class DepthLocationCode<uint16_t, uint16_t>;
template class DepthLocationCode<uint16_t, uint32_t>;
template class DepthLocationCode<uint16_t, uint64_t>;

template class DepthLocationCode<uint32_t, uint8_t>;
template class DepthLocationCode<uint32_t, uint16_t>;
template class DepthLocationCode<uint32_t, uint32_t>;
template class DepthLocationCode<uint32_t, uint64_t>;

template class DepthLocationCode<uint64_t, uint8_t>;
template class DepthLocationCode<uint64_t, uint16_t>;
template class DepthLocationCode<uint64_t, uint32_t>;
template class DepthLocationCode<uint64_t, uint64_t>;

} // namespace Voxoctis
