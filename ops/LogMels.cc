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

#include "LogMels.h"

#include <glog/logging.h>
#include <complex>

const int kAudioBufferSize = 20480;
const double kGain = 1080674186.3482928;

inline float hz2Mel(float hz) {
  return 1127 * std::log(1 + hz / 700);
}

inline float mel2Hz(float mel) {
  return 700 * (std::exp(mel / 1127) - 1);
}

LogSpectrum::LogSpectrum(
    int numChannels,
    float samplingRate,
    int windowLength,
    int windowStep,
    float startFreq,
    float endFreq,
    bool normalized)
    : numChannels_(numChannels),
      samplingRate_(samplingRate),
      windowLength_(windowLength * samplingRate / 1000.0f),
      windowStep_(windowStep * samplingRate / 1000.0f),
      startFreq_(startFreq),
      endFreq_(endFreq > 0 ? endFreq : samplingRate * .5f),
      normalized_(normalized),
      fifo_(nullptr),
      inited_(false) {
  int bits = log2(windowLength_);
  if (pow(2, bits) != windowLength_) {
    bits += 1;
  }
  fftSize_ = (int)pow(2, bits);
  filterLength_ = fftSize_ / 2 + 1;

  filters_ = (float*)calloc(numChannels_ * filterLength_, sizeof(float));
  featFrame_ = (double*)calloc(numChannels_, sizeof(double));
  tempFrame_ = (float*)calloc(fftSize_, sizeof(float));
  window_ = (float*)calloc(windowLength_, sizeof(float));

  dftContext_ = av_rdft_init(bits, DFT_R2C);
  for (int i = 0; i < windowLength_; i++) {
    window_[i] = .5f * (1 - cos(2 * M_PI * i / (windowLength_ - 1)));
  }

  CHECK(Init());
}

void LogSpectrum::Cleanup() {
  if (filters_) {
    free(filters_);
    filters_ = nullptr;
  }
  if (featFrame_) {
    free(featFrame_);
    featFrame_ = nullptr;
  }
  if (tempFrame_) {
    free(tempFrame_);
    tempFrame_ = nullptr;
  }
  if (window_) {
    free(window_);
    window_ = nullptr;
  }
  if (dftContext_) {
    av_rdft_end(dftContext_);
    dftContext_ = nullptr;
  }
  if (fifo_) {
    av_audio_fifo_reset(fifo_);
    av_audio_fifo_free(fifo_);
    fifo_ = nullptr;
  }
}

LogSpectrum::~LogSpectrum() {
  Cleanup();
}

bool LogSpectrum::Init() {
  if (inited_) {
    return true;
  }
  if (!fifo_) {
    if (!(fifo_ =
              av_audio_fifo_alloc(AV_SAMPLE_FMT_FLT, 1, kAudioBufferSize))) {
      LOG(ERROR) << "Could not allocate FIFO";
      return false;
    }
  }

  float maxFreq = samplingRate_ / 2.0f;
  CHECK_LT(startFreq_, endFreq_) << "End frequency is larger than start freq";
  CHECK_LE(endFreq_, maxFreq) << "End frequency is larger than maxFreq";

  float startMel = hz2Mel(startFreq_);
  float endMel = hz2Mel(endFreq_);
  float dx = (endMel - startMel) / (numChannels_ + 1);
  float freqStep = samplingRate_ / 2 / filterLength_;
  for (int filter = 0; filter < numChannels_; ++filter) {
    float start = mel2Hz(filter * dx) / freqStep;
    float mid = mel2Hz((filter + 1) * dx) / freqStep;
    float end = mel2Hz((filter + 2) * dx) / freqStep;
    int from = int(start);
    int to = int(end) + 1;
    float leftWidth = std::max<float>(1.0, mid - start);
    float rightWidth = std::max<float>(1.0, end - mid);
    float sum = 0;
    from = std::max<int>(0, from);
    to = std::min<int>(filterLength_ - 1, to);
    filterRanges_.emplace_back(from, to);
    for (int i = from; i <= to; i++) {
      float value =
          1.f - ((i < mid) ? (mid - i) / leftWidth : (i - mid) / rightWidth);
      if (value > 0) {
        filters_[filter * filterLength_ + i] = value;
        sum += value;
      }
    }
    if (normalized_ && sum > 0) {
      for (int inX = from; inX <= to; inX++) {
        filters_[filter * filterLength_ + inX] /= sum;
      }
    }
  }
  inited_ = true;
  return true;
}

int LogSpectrum::Write(float* samples, size_t numSamples) {
  int samplesW = 0;
  if ((samplesW = av_audio_fifo_write(fifo_, (void**)&samples, numSamples)) <
      numSamples) {
    LOG(ERROR) << "Could not write data to FIFO";
    return -1;
  }
  return samplesW;
}

int LogSpectrum::Read(float* feat) {
  if (av_audio_fifo_size(fifo_) > windowLength_) {
    memset(tempFrame_, 0, fftSize_ * sizeof(float));
    if (av_audio_fifo_peek(fifo_, (void**)&tempFrame_, windowLength_) <
        windowLength_) {
      LOG(ERROR) << "Could not read data from FIFO";
      return -1;
    }
    if (av_audio_fifo_drain(fifo_, windowStep_)) {
      LOG(ERROR) << "Could not drain data from FIFO";
      return -1;
    }
    Apply(feat);
    return 1;
  }
  return 0;
}

void LogSpectrum::Apply(float* feat) {
  PowerSpectrum();
  for (int i = 0; i < numChannels_; ++i) {
    featFrame_[i] = 0;
  }
  for (int i = 0; i < numChannels_; ++i) {
    for (int j = filterRanges_[i].first; j < filterRanges_[i].second; ++j) {
      featFrame_[i] += tempFrame_[j] * filters_[i * filterLength_ + j] * kGain;
    }
    feat[i] =
        featFrame_[i] > M_E ? std::log(featFrame_[i]) : featFrame_[i] / M_E;
  }
}

void LogSpectrum::PowerSpectrum() {
  for (int i = 0; i < windowLength_; i++) {
    tempFrame_[i] *= window_[i];
  }
  av_rdft_calc(dftContext_, tempFrame_);
  FFTComplex* comps = reinterpret_cast<FFTComplex*>(tempFrame_);
  for (int i = 0; i < fftSize_ / 2; i++) {
    tempFrame_[i] = comps[i].re * comps[i].re + comps[i].im * comps[i].im;
  }
}
