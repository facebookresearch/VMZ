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

#ifndef LOGMELS_H
#define LOGMELS_H

#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

#define __STDC_CONSTANT_MACROS
extern "C" {
#include <libavcodec/avfft.h>
#include <libavutil/audio_fifo.h>
}


class LogSpectrum {
 public:
  LogSpectrum(
      int numChannels,
      float samplingRate,
      int windowLength,
      int windowStep,
      float startFreq = 0,
      float endFreq = -1,
      bool normalized = true);

  ~LogSpectrum();

  void Cleanup();

  bool Init();

  int Write(float* first, size_t numSamples);

  int Read(float* feat);

 private:
  void Apply(float* feat);

  void PowerSpectrum();

  int numChannels_;
  float samplingRate_;
  int windowLength_;
  int windowStep_;
  int fftSize_;
  float startFreq_;
  float endFreq_;
  bool normalized_;

  int filterLength_;
  float* filters_;
  double* featFrame_;
  float* tempFrame_;
  std::vector<std::pair<int, int>> filterRanges_;

  RDFTContext* dftContext_;
  float_t* window_;
  AVAudioFifo* fifo_;
  bool inited_;
};


#endif // LOGMELS_H
