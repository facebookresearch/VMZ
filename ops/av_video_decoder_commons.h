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

/*
 * common structs shared by av decoder and original Caffe2 video decoder
 */

#ifndef AV_VIDEO_DECODER_COMMONS_H_
#define AV_VIDEO_DECODER_COMMONS_H_

#include <stdio.h>
#include <memory>
#include <string>
#include <vector>
#include "caffe2/core/logging.h"
// #include "common/time/Time.h"
// #include "common/base/Exception.h"
// #include <folly/ScopeGuard.h>
// #include <folly/Format.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
// #include <libavutil/audioconvert.h>
#include <libavutil/log.h>
#include <libavutil/motion_vector.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

namespace caffe2 {

namespace video_modeling {

#define VIO_BUFFER_SZ 32768
#define MAX_DECODING_FRAMES 10000

// enum to specify 3 special fps sampling behaviors:
// 0: disable fps sampling, no frame sampled at all
// -1: unlimited fps sampling, will sample at native video fps
// -2: disable fps sampling, but will get the frame at specific timestamp
enum SpecialFps {
  SAMPLE_NO_FRAME = 0,
  SAMPLE_ALL_FRAMES = -1,
  SAMPLE_TIMESTAMP_ONLY = -2,
};

// three different types of resolution when decoding the video
// 0: resize to width x height and ignore the aspect ratio;
// 1: resize to short_edge and keep the aspect ratio;
// 2: using the original resolution of the video; if resolution
//    is smaller than crop_size x crop_size, resize to crop_size
//    and keep the aspect ratio;
enum VideoResType {
  USE_WIDTH_HEIGHT = 0,
  USE_SHORT_EDGE = 1,
  ORIGINAL_RES = 2,
};

// three different types of decoding behavior are supported
// 0: do temporal jittering to sample a random clip from the video
// 1: sample a clip from a given starting frame
// 2: uniformly sample multiple clips from the video;
enum DecodeType {
  DO_TMP_JITTER = 0,
  DO_UNIFORM_SMP = 1,
  USE_START_FRM = 2,
};

// sampling interval for fps starting at specified timestamp
// use enum SpecialFps to set special fps decoding behavior
// note sampled fps will not always accurately follow the target fps,
// because sampled frame has to snap to actual frame timestamp,
// e.g. video fps = 25, sample fps = 4 will sample every 0.28s, not 0.25
// video fps = 25, sample fps = 5 will sample every 0.24s, not 0.2,
// because of floating-point division accuracy (1 / 5.0 is not exactly 0.2)
struct SampleInterval {
  double timestamp;
  double fps;
  SampleInterval() : timestamp(-1), fps(SpecialFps::SAMPLE_ALL_FRAMES) {}
  SampleInterval(double ts, double f) : timestamp(ts), fps(f) {}
  bool operator<(const SampleInterval& itvl) const {
    return (timestamp < itvl.timestamp);
  }
};

/// data structure for storing decoded video frames
class DecodedFrame {
 public:
  struct avDeleter {
    void operator()(unsigned char* p) const {
      av_free(p);
    }
  };
  using AvDataPtr = std::unique_ptr<uint8_t, avDeleter>;

  // decoded data buffer
  AvDataPtr data_;

  // size in bytes
  int size_ = 0;

  // frame dimensions
  int width_ = 0;
  int height_ = 0;

  // timestamp in seconds since beginning of video
  double timestamp_ = 0;

  // true if this is a key frame.
  bool keyFrame_ = false;

  // index of frame in video
  int index_ = -1;

  // Sequential number of outputted frame
  int outputFrameIndex_ = -1;

  // Microseconds it took to read and decode frame
  int64_t frameDecodeTimeUS_ = 0;
};

// data structure for storing decoded audio data
struct DecodedAudio {
  int dataSize_;
  int outSampleSize_;
  std::unique_ptr<float[]> audio_data_;

  explicit DecodedAudio(
      int dataSize = 0,
      int outSampleSize = 0,
      std::unique_ptr<float[]> audio_data = nullptr)
      : dataSize_(dataSize),
        outSampleSize_(outSampleSize),
        audio_data_(std::move(audio_data)){}
};

struct VideoMeta {
  double fps;
  int width;
  int height;
  enum AVMediaType codec_type;
  AVPixelFormat pixFormat;
  VideoMeta()
      : fps(-1),
        width(-1),
        height(-1),
        codec_type(AVMEDIA_TYPE_VIDEO),
        pixFormat(AVPixelFormat::AV_PIX_FMT_RGB24) {}
};

} // namespace video_modeling

} // namespace caffe2

#endif // AV_VIDEO_DECODER_COMMONS_H_
