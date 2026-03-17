#pragma once

#include <vector>

namespace vinput::audio {

// Peak-normalize samples in-place. Only amplifies (never attenuates).
// If the peak amplitude is below `low_threshold`, scales so peak == `target_peak`.
void PeakNormalize(std::vector<float> &samples, float target_peak = 1.0f,
                   float low_threshold = 0.1f);

}  // namespace vinput::audio
