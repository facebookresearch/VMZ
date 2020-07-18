/**
 * Copyright (c) 2020-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef AV_IO_H_
#define AV_IO_H_

#include <opencv2/opencv.hpp>

#include <random>
// #include "caffe/proto/caffe.pb.h"
#include "av_video_decoder_commons.h"
#include "LogMels.h"

#include <istream>
#include <ostream>

namespace caffe2 {

namespace video_modeling {

void ClipTransformRGB(
    const unsigned char* buffer_rgb,
    const int crop_size,
    const int length_rgb,
    const int channels_rgb,
    const int sampling_rate_rgb,
    const int height,
    const int width,
    const int h_off,
    const int w_off,
    const bool mirror_me,
    const std::vector<float>& mean_rgb,
    const std::vector<float>& inv_std_rgb,
    float* transformed_clip);

void ClipTransformAudioLogmel(
    const int decode_type_,
    const bool get_rgb_,
    const float* clip_rgb_data,
    const int number_of_frames,
    const bool tune_audio_step_,
    const int logMelFrames_,
    const int logMelAudioSamplingRate_,
    const int logMelWindowSizeMs_,
    const int logMelWindowStepMs_,
    const int logMelFilters_,
    const int num_of_required_frame_,
    const int align_audio_,
    const int clip_per_video_,
    const int audio_length_,
    const int clip_start_frame,
    std::vector<float> audioSamples,
    float* clip_of_logmels_data
);

} // namespace video_modeling

} // namespace caffe2

#endif // AV_IO_H_
