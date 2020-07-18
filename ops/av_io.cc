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

#include "av_io.h"
#include <random>
#include <string>
#include <algorithm>
#include "caffe2/core/logging.h"
#include <cstdlib>

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
    float* transformed_clip) {
  // The order of output dimensions is C, L, H, W
  int orig_index, tran_index;
  for (int c = 0; c < channels_rgb; ++c) {
    for (int l = 0; l < length_rgb; ++l) {
      int orig_index_l = l * sampling_rate_rgb * height * width * channels_rgb;
      int tran_index_l = (c * length_rgb + l) * crop_size;

      for (int h = 0; h < crop_size; ++h) {
        int orig_index_h = orig_index_l + (h + h_off) * width * channels_rgb;
        int tran_index_h = (tran_index_l + h) * crop_size;

        for (int w = 0; w < crop_size; ++w) {
          orig_index = orig_index_h + (w + w_off) * channels_rgb + c;

          // mirror the frame
          if (mirror_me) {
            tran_index = tran_index_h + (crop_size - 1 - w);
          } else {
            tran_index = tran_index_h + w;
          }

          // normalize and transform the clip
          transformed_clip[tran_index] = (buffer_rgb[orig_index] - mean_rgb[c])
                                       * inv_std_rgb[c];
        }
      }
    }
  }
}

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
) {
  int newlogMelWindowStepMs_ = logMelWindowStepMs_;
  int newAudioLength_ = audio_length_;
  int newClipStartFrame_ = clip_start_frame;

  if (get_rgb_ && clip_rgb_data && number_of_frames && tune_audio_step_) {
    float framesNeed =
        number_of_frames * logMelFrames_ / num_of_required_frame_;
    float adjInterval =
      logMelWindowSizeMs_ * logMelAudioSamplingRate_ / 1000.0f;
    float newAdjStep =
      (audioSamples.size() - adjInterval) / (framesNeed - 1);
    int newStep =
      std::ceil(newAdjStep * 1000.0f / logMelAudioSamplingRate_);
    newlogMelWindowStepMs_ = std::max(newStep, 1);
  }
  LogSpectrum spectrum(
      logMelFilters_,
      logMelAudioSamplingRate_,
      logMelWindowSizeMs_,
      newlogMelWindowStepMs_
  );
  spectrum.Write(audioSamples.data(), audioSamples.size());
  int framesRead = 0;
  vector<vector<float>> logmels;
  while(true){
    vector<float> buffer(logMelFilters_);
    if (spectrum.Read(buffer.data()) <= 0){
      break;
    }
    framesRead++;
    logmels.push_back(buffer);
  }

  if (get_rgb_ && clip_rgb_data && number_of_frames &&
    framesRead && align_audio_) {
    if (align_audio_ > 1 && newAudioLength_ == 0) {
      if (clip_per_video_ == 1 && framesRead >= logMelFrames_) {
        int audio_start_frame = std::floor(
          newClipStartFrame_ * framesRead / number_of_frames);
        int audio_left_start = audio_start_frame;
        if (align_audio_ == 3) { // align by center
          int clip_end_frame = newClipStartFrame_ + num_of_required_frame_;
          int audio_end_frame = std::ceil(
            clip_end_frame * framesRead / number_of_frames);
          int audio_middle_frame = std::round(
            (audio_start_frame + audio_end_frame) / 2);
          audio_left_start =
            std::max<int>(audio_middle_frame - std::ceil(logMelFrames_ / 2), 0);
        }
        int audio_right_end =
          std::min(audio_left_start + logMelFrames_, framesRead);
        logmels.erase(logmels.begin() + audio_right_end, logmels.end());
        if (audio_left_start > 0 &&
          audio_left_start < static_cast<int>(logmels.size())) {
          logmels.erase(
            logmels.begin(), logmels.begin() + audio_left_start);
        }
        framesRead = static_cast<int>(logmels.size());
      }
    } else {
      cv::Mat1f logmel_cv(framesRead, logMelFilters_);
      cv::Mat1f logmel_interpolated;
      if (align_audio_ == 1) { // perfect align
        newAudioLength_ = num_of_required_frame_;
      }
      framesRead = std::ceil(
        number_of_frames * logMelFrames_ / newAudioLength_);

      for (int i = 0; i < logmel_cv.rows; ++i) {
        for (int j = 0; j < logmel_cv.cols; ++j) {
          logmel_cv.at<float>(i,j) = logmels[i][j];
        }
      }

      if (framesRead >= logmel_cv.rows) {
        cv::resize(logmel_cv, logmel_interpolated,
            cv::Size(logMelFilters_, framesRead));
      } else {
        cv::resize(logmel_cv, logmel_interpolated,
            cv::Size(logMelFilters_, framesRead), 0, 0,
            cv::INTER_NEAREST);
      }
      logmels.clear();

      if (decode_type_ == DecodeType::DO_UNIFORM_SMP) {
        for(int i = 0; i < logmel_interpolated.rows; ++i) {
          vector<float> buffer(logMelFilters_);
          for (int j = 0; j < logmel_interpolated.cols; ++j) {
            buffer[j] = logmel_interpolated.at<float>(i,j);
          }
          logmels.push_back(buffer);
        }
      } else {
        if (newAudioLength_ != num_of_required_frame_ &&
          align_audio_ == 3) { // center align
          int clip_middle_frame =
            std::round(newClipStartFrame_ + num_of_required_frame_ / 2);
          newClipStartFrame_ =
            std::max(clip_middle_frame - newAudioLength_ / 2, 0);
        }
        int audio_start_frame = std::floor(
          newClipStartFrame_ * framesRead / number_of_frames);
        int audio_end_frame = std::min(
          logmel_interpolated.rows, audio_start_frame + logMelFrames_);
        for(int i = audio_start_frame; i < audio_end_frame; ++i) {
          vector<float> buffer(logMelFilters_);
          for (int j = 0; j < logmel_interpolated.cols; ++j) {
            buffer[j] = logmel_interpolated.at<float>(i,j);
          }
          logmels.push_back(buffer);
        }
      }
      framesRead = static_cast<int>(logmels.size());
    }
    // pad during testing; disable for now due to poor performance;
    // if (decode_type_ == DecodeType::DO_UNIFORM_SMP &&
    //   align_audio_ > 1 && clip_per_video_ > 1) {
    //   int audio_pad_difference = 0;
    //   int aligned_audio_length = std::ceil(
    //     num_of_required_frame_ * framesRead / number_of_frames);
    //   if (logMelFrames_ > aligned_audio_length) {
    //     audio_pad_difference = logMelFrames_ - aligned_audio_length;
    //   }
    //   if (align_audio_ == 3) {
    //     vector<float> pad(logMelFilters_);
    //     logmels.insert(
    //       logmels.begin(), std::floor(audio_pad_difference / 2), pad);
    //     logmels.insert(
    //       logmels.end(), std::ceil(audio_pad_difference / 2), pad);
    //   }
    //   if (align_audio_ == 2) {
    //     vector<float> pad(logMelFilters_);
    //     logmels.insert(
    //       logmels.end(), audio_pad_difference, pad);
    //   }
    // }
  }

  float audio_pad_difference = 0.0;
  if (decode_type_ == DecodeType::DO_UNIFORM_SMP &&
    align_audio_ > 1 && clip_per_video_ > 1 && number_of_frames) {
    float aligned_audio_length =
      num_of_required_frame_ * framesRead / number_of_frames;
    audio_pad_difference = logMelFrames_ - aligned_audio_length;
  }
  const float frameStep = std::max<float>(
    0, (framesRead + audio_pad_difference - logMelFrames_))
    / std::max<int>(1, clip_per_video_ - 1);
  const int clipSize = logMelFilters_ * logMelFrames_;
  for (int i = 0; i < clip_per_video_; ++i){
    float* clip_of_logmels_data_start = clip_of_logmels_data + i * clipSize;
    memset(clip_of_logmels_data_start, 0,
        clipSize * sizeof(float));
    float start_left_shift = 0.0;
    if (align_audio_ > 1 && clip_per_video_ > 1
      && i * frameStep + logMelFrames_ > framesRead) {
      // last frame, uniform sampling, not perfect alignment, shift by the
      // audio_pad_difference
      start_left_shift = i * frameStep + logMelFrames_ - framesRead;
    } else if (align_audio_ == 3 && clip_per_video_ > 1) {
      // align by center, every frame needs to shift by half of pad diff
      start_left_shift = audio_pad_difference / 2;
    }
    const int startFrame =
        std::max<int>(std::floor(i * frameStep - start_left_shift), 0);
    const int endFrame =
        std::min<int>(startFrame + logMelFrames_, framesRead);
    for (int j = startFrame; j < endFrame; ++j){
      memcpy(clip_of_logmels_data_start + (j - startFrame) * logMelFilters_,
          logmels[j].data(), logMelFilters_ * sizeof(float));
    }
    VLOG(2) << "Copied data " << (startFrame) << " to " << endFrame
            << " total frames " << framesRead << " index " << i
            << " out of " << clip_per_video_;
  }
}

} // namespace video_modeling

} // namespace caffe2
