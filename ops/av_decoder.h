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
 * AV decoder for decoding both audio and visual signals
 */

#ifndef AV_DECODER_H_
#define AV_DECODER_H_

#include <stdio.h>
#include <memory>
#include <string>
#include <vector>
#include "caffe2/core/logging.h"
// #include "common/time/Time.h"
// #include "common/base/Exception.h"
// #include <folly/ScopeGuard.h>
// #include <folly/Format.h>
#include "av_video_decoder_commons.h"

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

class Params {
 public:
  // return all key-frames regardless of specified fps
  bool keyFrames_ = false;

  // return video data while decoding  media
  bool getVideo_ = false;

  // return audio data while decoding  media
  bool getAudio_ = false;

  // for sampling audio data
  int outrate_ = 16000;
  AVSampleFormat outfmt_ = AV_SAMPLE_FMT_FLT;
  int64_t outlayout_ = AV_CH_LAYOUT_MONO;

  // Output image pixel format
  AVPixelFormat pixelFormat_ = AVPixelFormat::AV_PIX_FMT_RGB24;

  // Index of stream to decode.
  // -1 will automatically decode the first video stream.
  int streamIndex_ = -1;

  // How many frames to output at most from the video
  // -1 no limit
  int maximumOutputFrames_ = -1;

  // params for video resolution
  int video_res_type_ = VideoResType::USE_WIDTH_HEIGHT;
  int crop_size_ = -1;
  int short_edge_ = -1;

  // Output video size, -1 to preserve origianl dimension
  int outputWidth_ = -1;
  int outputHeight_ = -1;

  // max output dimension, -1 to preserve original size
  // the larger dimension of the video will be scaled to this size,
  // and the second dimension will be scaled to preserve aspect ratio
  int maxOutputDimension_ = -1;

  // params for decoding behavior
  int decode_type_ = DecodeType::DO_TMP_JITTER;
  int num_of_required_frame_ = -1;

  // intervals_ control variable sampling fps between different timestamps
  // intervals_ must be ordered strictly ascending by timestamps
  // the first interval must have a timestamp of zero
  // fps must be either the 3 special fps defined in SpecialFps, or > 0
  std::vector<SampleInterval> intervals_ = {{0, SpecialFps::SAMPLE_ALL_FRAMES}};

  Params() {}

  /**
   * FPS of output frames
   * setting here will reset intervals_ and force decoding at target FPS
   * This can be used if user just want to decode at a steady fps
   */
  Params& fps(float v) {
    intervals_.clear();
    intervals_.emplace_back(0, v);
    return *this;
  }

  /**
   * Sample output frames at a specified list of timestamps
   * Timestamps must be in increasing order, and timestamps past the end of the
   * video will be ignored
   * Setting here will reset intervals_
   */
  Params& setSampleTimestamps(const std::vector<double>& timestamps) {
    intervals_.clear();
    // insert an interval per desired frame.
    for (auto& timestamp : timestamps) {
      intervals_.emplace_back(timestamp, SpecialFps::SAMPLE_TIMESTAMP_ONLY);
    }
    return *this;
  }

  /**
   * Pixel format of output buffer, default PIX_FMT_RGB24
   */
  Params& pixelFormat(AVPixelFormat pixelFormat) {
    pixelFormat_ = pixelFormat;
    return *this;
  }

  /**
   * Return all key-frames
   */
  Params& keyFrames(bool keyFrames) {
    keyFrames_ = keyFrames;
    return *this;
  }

  /**
   * Index of video stream to process, defaults to the first video stream
   */
  Params& streamIndex(int index) {
    streamIndex_ = index;
    return *this;
  }

  /**
   * Only output this many frames, default to no limit
   */
  Params& maxOutputFrames(int count) {
    maximumOutputFrames_ = count;
    return *this;
  }

  /**
   * Output frame width, default to video width
   */
  Params& outputWidth(int width) {
    outputWidth_ = width;
    return *this;
  }

  /**
   * Output frame height, default to video height
   */
  Params& outputHeight(int height) {
    outputHeight_ = height;
    return *this;
  }

  /**
   * Max dimension of either width or height, if any is bigger
   * it will be scaled down to this and econd dimension
   * will be scaled down to maintain aspect ratio.
   */
  Params& maxOutputDimension(int size) {
    maxOutputDimension_ = size;
    return *this;
  }
};

class VideoIOContext {
 public:
  explicit VideoIOContext(const std::string& fname)
      : workBuffersize_(VIO_BUFFER_SZ),
        workBuffer_((uint8_t*)av_malloc(workBuffersize_)),
        inputFile_(nullptr),
        inputBuffer_(nullptr),
        inputBufferSize_(0) {
    inputFile_ = fopen(fname.c_str(), "rb");
    if (inputFile_ == nullptr) {
        LOG(ERROR) << "Error opening video file " << fname;
    }
    ctx_ = avio_alloc_context(
        static_cast<unsigned char*>(workBuffer_.get()),
        workBuffersize_,
        0,
        this,
        &VideoIOContext::readFile,
        nullptr, // no write function
        &VideoIOContext::seekFile);
  }

  explicit VideoIOContext(const char* buffer, int size)
      : workBuffersize_(VIO_BUFFER_SZ),
        workBuffer_((uint8_t*)av_malloc(workBuffersize_)),
        inputFile_(nullptr),
        inputBuffer_(buffer),
        inputBufferSize_(size) {
    ctx_ = avio_alloc_context(
        static_cast<unsigned char*>(workBuffer_.get()),
        workBuffersize_,
        0,
        this,
        &VideoIOContext::readMemory,
        nullptr, // no write function
        &VideoIOContext::seekMemory);
  }

  ~VideoIOContext() {
    av_free(ctx_);
    if (inputFile_) {
      fclose(inputFile_);
    }
  }

  int read(unsigned char* buf, int buf_size) {
    if (inputBuffer_) {
      return readMemory(this, buf, buf_size);
    } else if (inputFile_) {
      return readFile(this, buf, buf_size);
    } else {
      return -1;
    }
  }

  int64_t seek(int64_t offset, int whence) {
    if (inputBuffer_) {
      return seekMemory(this, offset, whence);
    } else if (inputFile_) {
      return seekFile(this, offset, whence);
    } else {
      return -1;
    }
  }

  static int readFile(void* opaque, unsigned char* buf, int buf_size) {
    VideoIOContext* h = static_cast<VideoIOContext*>(opaque);
    if (feof(h->inputFile_)) {
      return AVERROR_EOF;
    }
    size_t ret = fread(buf, 1, buf_size, h->inputFile_);
    if (ret < buf_size) {
      if (ferror(h->inputFile_)) {
        return -1;
      }
    }
    return ret;
  }

  static int64_t seekFile(void* opaque, int64_t offset, int whence) {
    VideoIOContext* h = static_cast<VideoIOContext*>(opaque);
    switch (whence) {
      case SEEK_CUR: // from current position
      case SEEK_END: // from eof
      case SEEK_SET: // from beginning of file
        return fseek(h->inputFile_, static_cast<long>(offset), whence);
        break;
      case AVSEEK_SIZE:
        int64_t cur = ftell(h->inputFile_);
        fseek(h->inputFile_, 0L, SEEK_END);
        int64_t size = ftell(h->inputFile_);
        fseek(h->inputFile_, cur, SEEK_SET);
        return size;
    }

    return -1;
  }

  static int readMemory(void* opaque, unsigned char* buf, int buf_size) {
    VideoIOContext* h = static_cast<VideoIOContext*>(opaque);
    if (buf_size < 0) {
      return -1;
    }

    int reminder = h->inputBufferSize_ - h->offset_;
    int r = buf_size < reminder ? buf_size : reminder;
    if (r < 0) {
      return AVERROR_EOF;
    }

    memcpy(buf, h->inputBuffer_ + h->offset_, r);
    h->offset_ += r;
    return r;
  }

  static int64_t seekMemory(void* opaque, int64_t offset, int whence) {
    VideoIOContext* h = static_cast<VideoIOContext*>(opaque);
    switch (whence) {
      case SEEK_CUR: // from current position
        h->offset_ += offset;
        break;
      case SEEK_END: // from eof
        h->offset_ = h->inputBufferSize_ + offset;
        break;
      case SEEK_SET: // from beginning of file
        h->offset_ = offset;
        break;
      case AVSEEK_SIZE:
        return h->inputBufferSize_;
    }
    return h->offset_;
  }

  AVIOContext* get_avio() {
    return ctx_;
  }

 private:
  int workBuffersize_;
  DecodedFrame::AvDataPtr workBuffer_;
  // for file mode
  FILE* inputFile_;

  // for memory mode
  const char* inputBuffer_;
  int inputBufferSize_;
  int offset_ = 0;

  AVIOContext* ctx_;
};

class Callback {
 public:
  virtual void frameDecoded(std::unique_ptr<DecodedFrame> img) = 0;
  virtual void audioDecoded(float* /*samples*/, int /*numSamples*/) {}
  virtual void videoDecodingStarted(const VideoMeta& /*videoMeta*/) {}
  virtual void videoDecodingEnded(double /*lastFrameTimestamp*/) {}
  virtual ~Callback() {}
};

class AVDecoder {
 public:
  AVDecoder();

  void decodeFile(
      const std::string& filename,
      const Params& params,
      const int start_frm,
      Callback& callback,
      int& number_of_frames);

  void decodeMemory(
      const std::string& filename,
      const char* buffer,
      const int size,
      const Params& params,
      const int start_frm,
      Callback& callback,
      int& number_of_frames);

 private:
  std::string ffmpegErrorStr(int result);

  void ResizeAndKeepAspectRatio(
      const int origWidth,
      const int origHeight,
      const int short_edge,
      const int long_edge,
      int& outWidth,
      int& outHeight);

  void getAudioSample(
      AVPacket& packet,
      AVCodecContext* audioCodecContext_,
      AVFrame* audioStreamFrame_,
      SwrContext* convertCtx_,
      Callback& callback,
      const Params& params);

  void decodeLoop(
      const std::string& videoName,
      VideoIOContext& ioctx,
      const Params& params,
      const int start_frm,
      Callback& callback,
      int& number_of_frames);
};

void FreeAVDecodedData(
    std::vector<std::unique_ptr<DecodedFrame>>& sampledFrames,
    std::vector<float>& sampledAudio);

bool DecodeMultipleAVClipsFromVideo(
    const char* video_buffer,
    const std::string& video_filename,
    const int encoded_size,
    const Params& params,
    const int start_frm,
    const int clip_per_video,
    const bool use_local_file,
    int& height,
    int& width,
    std::vector<unsigned char*>& buffer_rgb,
    std::vector<float>& buffer_audio,
    int& number_of_frames,
    int& clip_start_frame
);

class CallbackImpl : public Callback {
public:
  std::vector<std::unique_ptr<DecodedFrame>> frames;
  std::vector<float> audio_samples;

  explicit CallbackImpl () {
    clear();
  }

  void clear() {
    FreeAVDecodedData(frames, audio_samples);
  }

  void frameDecoded(std::unique_ptr<DecodedFrame> frame) override {
    frames.push_back(move(frame));
  }

  void audioDecoded(float* samples, int numSamples) override {
    for (int i = 0; i < numSamples; ++i){
      audio_samples.push_back(samples[i]);
    }
  }

  void videoDecodingStarted(const VideoMeta& /*videoMeta*/) override {
    clear();
  }
};

} // namespace video_modeling

} // namespace caffe2

#endif // AV_DECODER_H_
