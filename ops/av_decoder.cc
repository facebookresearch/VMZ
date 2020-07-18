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

#include "av_decoder.h"
#include <mutex>
#include <random>

namespace caffe2 {

namespace video_modeling {

namespace {
const int AUDIO_INBUF_SIZE = 20480;
const int64_t kAudioMargin = 1000000; //TODO move hardcode to parameters

}

AVDecoder::AVDecoder() {
  static bool gInitialized = false;
  static std::mutex gMutex;
  std::unique_lock<std::mutex> lock(gMutex);
  if (!gInitialized) {
    av_register_all();
    avcodec_register_all();
    avformat_network_init();
    gInitialized = true;
  }
}

void AVDecoder::getAudioSample(
    AVPacket& packet,
    AVCodecContext* audioCodecContext_,
    AVFrame* audioStreamFrame_,
    SwrContext* convertCtx_,
    Callback& callback,
    const Params& params) {
  int frame_finished = 0;
  auto result = avcodec_decode_audio4(
      audioCodecContext_, audioStreamFrame_, &frame_finished, &packet);

  if (frame_finished) {
    int error;
    uint8_t* convertedInputSamples = nullptr;
    auto c = audioCodecContext_;
    if (!(convertedInputSamples = (uint8_t*) calloc(c->channels,
        sizeof(convertedInputSamples)))
        ) {
      LOG(FATAL) << "Could not allocate converted input sample pointers";
    }

    if ((error = av_samples_alloc(
        &convertedInputSamples,
        nullptr,
        c->channels,
        AUDIO_INBUF_SIZE,
        params.outfmt_, 0)) < 0) {
      LOG(ERROR) << "Could not allocate converted input samples. error code '"
                 << error << "'";
    }

    int numberOfOutputSamples;
    if ((numberOfOutputSamples = swr_convert(
        convertCtx_,
        &convertedInputSamples,
        AUDIO_INBUF_SIZE,
        (const uint8_t**)&audioStreamFrame_->extended_data[0],
        audioStreamFrame_->nb_samples)) < 0) {
      LOG(ERROR) << "Could not convert input samples. error '"
                 << numberOfOutputSamples << "'";
    }
    if (numberOfOutputSamples > 0 && convertedInputSamples != nullptr){
      callback.audioDecoded(
          (float*)convertedInputSamples, numberOfOutputSamples);
    }

    //Cleanup
    av_freep(&convertedInputSamples);
    free(convertedInputSamples);
  } else {
    result = packet.size;
  }
  packet.size -= result;
  packet.data += result;
}

void AVDecoder::ResizeAndKeepAspectRatio(
    const int origWidth,
    const int origHeight,
    const int short_edge,
    const int long_edge,
    int& outWidth,
    int& outHeight) {
  if (origWidth < origHeight) {
    // dominant height
    if (short_edge > 0) {
      // use short_edge for rescale
      float ratio = short_edge / float(origWidth);
      outWidth = short_edge;
      outHeight = (int)round(ratio * origHeight);
    } else {
      // use long_edge for rescale
      float ratio = long_edge / float(origHeight);
      outHeight = long_edge;
      outWidth = (int)round(ratio * origWidth);
    }
  } else {
    // dominant width
    if (short_edge > 0) {
      // use short_edge for rescale
      float ratio = short_edge / float(origHeight);
      outHeight = short_edge;
      outWidth = (int)round(ratio * origWidth);
    } else {
      // use long_edge for rescale
      float ratio = long_edge / float(origWidth);
      outWidth = long_edge;
      outHeight = (int)round(ratio * origHeight);
    }
  }
}

void AVDecoder::decodeLoop(
    const string& videoName,
    VideoIOContext& ioctx,
    const Params& params,
    const int start_frm,
    Callback& callback,
    int& number_of_frames) {
  AVPixelFormat pixFormat = params.pixelFormat_;
  AVFormatContext* inputContext = avformat_alloc_context();
  AVStream* videoStream_ = nullptr;
  AVStream* audioStream_ = nullptr;
  AVCodecContext* videoCodecContext_ = nullptr;
  AVCodecContext* audioCodecContext_ = nullptr;
  AVFrame* videoStreamFrame_ = nullptr;
  AVFrame* audioStreamFrame_ = nullptr;
  SwrContext* convertCtx_ = nullptr;
  AVPacket packet;
  av_init_packet(&packet); // init packet
  SwsContext* scaleContext_ = nullptr;

  try {
    inputContext->pb = ioctx.get_avio();
    inputContext->flags |= AVFMT_FLAG_CUSTOM_IO;
    int ret = 0;

    // Determining the input format:
    int probeSz = 1 * 1024 + AVPROBE_PADDING_SIZE;
    DecodedFrame::AvDataPtr probe((uint8_t*)av_malloc(probeSz));
    memset(probe.get(), 0, probeSz);
    int len = ioctx.read(probe.get(), probeSz - AVPROBE_PADDING_SIZE);
    if (len < probeSz - AVPROBE_PADDING_SIZE) {
      LOG(ERROR) << "Insufficient data to determine video format";
    }
    // seek back to start of stream
    ioctx.seek(0, SEEK_SET);

    unique_ptr<AVProbeData> probeData(new AVProbeData());
    probeData->buf = probe.get();
    probeData->buf_size = len;
    probeData->filename = "";
    // Determine the input-format:
    inputContext->iformat = av_probe_input_format(probeData.get(), 1);
    // this is to avoid the double-free error
    if (inputContext->iformat == nullptr) {
      LOG(ERROR) << "inputContext iformat is nullptr!";
    }
    ret = avformat_open_input(&inputContext, "", nullptr, nullptr);
    if (ret < 0) {
      LOG(ERROR) << "Unable to open stream : " << ffmpegErrorStr(ret);
      return;
    }

    ret = avformat_find_stream_info(inputContext, nullptr);
    if (ret < 0) {
      LOG(ERROR) << "Unable to find stream info in " << videoName << " "
                 << ffmpegErrorStr(ret);
      return;
    }

    // Decode the first video stream
    int videoStreamIndex_ = params.streamIndex_;
    int audioStreamIndex_ = params.streamIndex_;
    if (params.streamIndex_ == -1) {
      for (int i = 0; i < inputContext->nb_streams; i++) {
        auto stream = inputContext->streams[i];
        if (stream->codec->codec_type == AVMEDIA_TYPE_VIDEO &&
            videoStreamIndex_ == -1) {
          videoStreamIndex_ = i;
          videoStream_ = stream;
        } else if (
            stream->codec->codec_type == AVMEDIA_TYPE_AUDIO &&
                audioStreamIndex_ == -1) {
          audioStreamIndex_ = i;
          audioStream_ = stream;
        }
        if (videoStreamIndex_ != -1 && audioStreamIndex_ != -1) {
          break;
        }
      }
    }

    //Video or audio stream  can be empty, but not both at the same time
    const bool hasVideo = params.getVideo_ && videoStreamIndex_ >= 0;
    const bool hasAudio = params.getAudio_ && audioStreamIndex_ >= 0;
    if (!hasAudio && !hasVideo) {
      LOG(ERROR) << "Neither video nor audio stream are being decoded in "
          << videoName << " : " << ffmpegErrorStr(ret)
          << " params.getVideo_=" <<  params.getVideo_
          << " params.getAudio_=" <<  params.getAudio_
          << " videoStreamIndex=" << videoStreamIndex_
          << " audioStreamIndex=" << audioStreamIndex_;
      return;

    }

    // Initialize codec
    AVDictionary* opts = nullptr;
    if (params.getAudio_ && audioStreamIndex_ >= 0) {
      audioStreamFrame_ = av_frame_alloc();
      audioCodecContext_ = inputContext->streams[audioStreamIndex_]->codec;
      ret = avcodec_open2(
          audioCodecContext_,
          avcodec_find_decoder(audioCodecContext_->codec_id),
          nullptr);

      if (ret < 0) {
        const std::string codecName =
            audioCodecContext_->codec != nullptr
            && audioCodecContext_->codec->name != nullptr ?
            std::string(audioCodecContext_->codec->name) : "None";


        LOG(ERROR) << "Cannot open audio codec : " << codecName;
      }

      convertCtx_ = swr_alloc_set_opts(
          nullptr,
          params.outlayout_,
          params.outfmt_,
          params.outrate_,
          audioCodecContext_->channel_layout,
          audioCodecContext_->sample_fmt,
          audioCodecContext_->sample_rate,
          0,
          nullptr);

      if (convertCtx_ == nullptr) {
        LOG(ERROR) << ("Cannot setup sample format converter.");
      }
      if (swr_init(convertCtx_) < 0) {
        LOG(ERROR) << ("Cannot init sample format converter.");
      }
    }

    bool mustDecodeAll = false;
    auto itvlIter = params.intervals_.begin();
    double currFps = 0;
    // frame index in video stream
    int frameIndex = -1;
    // frame index of outputed frames
    int outputFrameIndex = -1;
    double lastFrameTimestamp = -1.0;
    double timestamp = -1.0;
    long int start_ts = -1;
    double prevTimestamp = 0;
    int outWidth = 0;
    int outHeight = 0;

    if (params.getVideo_ && videoStreamIndex_ >= 0) {
      videoCodecContext_ = videoStream_->codec;
      try {
        ret = avcodec_open2(
            videoCodecContext_,
            avcodec_find_decoder(videoCodecContext_->codec_id),
            &opts);
      } catch (const std::exception&) {
        LOG(ERROR) << ("Exception during open video codec");
      }

      if (ret < 0) {
      LOG(ERROR) << "Cannot open video codec : "
                 << videoCodecContext_->codec->name;
      return;
    }

      // Calculate if we need to rescale the frames
      const int origWidth = videoCodecContext_->width;
      const int origHeight = videoCodecContext_->height;
      outWidth = origWidth;
      outHeight = origHeight;

      if (params.video_res_type_ == VideoResType::ORIGINAL_RES) {
        // if the original resolution is too low,
        // make it at least the same size as crop_size_
        if (params.crop_size_ > origWidth || params.crop_size_ > origHeight) {
          ResizeAndKeepAspectRatio(
              origWidth,
              origHeight,
              params.crop_size_,
              -1,
              outWidth,
              outHeight);
        }
      } else if (params.video_res_type_ == VideoResType::USE_SHORT_EDGE) {
        // resize the image to the predefined
        // short_edge_ resolution while keep the aspect ratio
        ResizeAndKeepAspectRatio(
            origWidth, origHeight, params.short_edge_, -1, outWidth, outHeight);
      } else if (params.video_res_type_ == VideoResType::USE_WIDTH_HEIGHT) {
        // resize the image to the predefined
        // resolution and ignore the aspect ratio
        outWidth = params.outputWidth_;
        outHeight = params.outputHeight_;
      } else {
        LOG(ERROR) << "Unknown VideoResType: " << params.video_res_type_;
      }

      // Make sure that we have a valid format
      if (videoCodecContext_->pix_fmt == AV_PIX_FMT_NONE) {
        LOG(ERROR) << ("pixel format is not valid.");
      }

      // Create a scale context
      scaleContext_ = sws_getContext(
          videoCodecContext_->width,
          videoCodecContext_->height,
          videoCodecContext_->pix_fmt,
          outWidth,
          outHeight,
          pixFormat,
          SWS_FAST_BILINEAR,
          nullptr,
          nullptr,
          nullptr);

      // Getting video meta data
      VideoMeta videoMeta;
      videoMeta.codec_type = videoCodecContext_->codec_type;
      videoMeta.width = outWidth;
      videoMeta.height = outHeight;
      videoMeta.pixFormat = pixFormat;

      // avoid division by zero, code adapted from https://www.ffmpeg.org/doxygen/0.6/rational_8h-source.html
      if (videoStream_->avg_frame_rate.num == 0 || videoStream_->avg_frame_rate.den == 0) {
        LOG(ERROR) << ("Frame rate is wrong. No data found.");
      }
      videoMeta.fps = av_q2d(videoStream_->avg_frame_rate);
      callback.videoDecodingStarted(videoMeta);
      number_of_frames = videoStream_->nb_frames;
      if (params.intervals_.size() == 0) {
        LOG(ERROR) << ("Empty sampling intervals.");
      }

      if (itvlIter->timestamp != 0) {
        LOG(ERROR) << ("Sampling interval starting timestamp is not zero.");
      }

      currFps = itvlIter->fps;
      if (currFps < 0 && currFps != SpecialFps::SAMPLE_ALL_FRAMES &&
          currFps != SpecialFps::SAMPLE_TIMESTAMP_ONLY) {
        // fps must be 0, -1, -2 or > 0
        LOG(ERROR) << ("Invalid sampling fps.");
      }

      prevTimestamp = itvlIter->timestamp;
      itvlIter++;
      if (itvlIter != params.intervals_.end() &&
          prevTimestamp >= itvlIter->timestamp) {
        LOG(ERROR) << ("Sampling interval timestamps must be strictly ascending.");
      }

      // Initialize frame and packet.
      // These will be reused across calls.
      videoStreamFrame_ = av_frame_alloc();
    }

    std::mt19937 meta_randgen(time(nullptr));
    /* identify the starting point from where we must start decoding */
    AVStream* stream = hasVideo ?  videoStream_ : audioStream_;
    const int64_t duration = av_rescale_q(stream->duration,
        stream->time_base, AV_TIME_BASE_Q);
    if ((videoStream_ != nullptr && videoStream_->duration > 0 &&
        videoStream_->nb_frames > 0) || (audioStream_ != nullptr &&
        audioStream_->duration > 0)) {
      /* we have a valid duration and nb_frames. We can safely
       * detect an intermediate timestamp to start decoding from. */

      // leave a margin of 10 frames to take in to account the error
      // from av_seek_frame
      long int margin = 0;
      if (hasVideo) {
        margin = int(ceil((10 * videoStream_->duration)
            / (videoStream_->nb_frames)));
      }
      // if we need to do temporal jittering
      if (params.decode_type_ == DecodeType::DO_TMP_JITTER) {
        /* estimate the average duration for the required # of frames */
        double maxFramesDuration = 0;
        if (hasVideo){
          maxFramesDuration =
              (videoStream_->duration * params.num_of_required_frame_) /
              (videoStream_->nb_frames);
        } else {
          maxFramesDuration = av_rescale_q(kAudioMargin, AV_TIME_BASE_Q,
              audioStream_->time_base);
        }
        int ts1 = 0;
        int ts2 = stream->duration - int(ceil(maxFramesDuration));
        ts2 = ts2 > 0 ? ts2 : 0;
        // pick a random timestamp between ts1 and ts2. ts2 is selected such
        // that you have enough frames to satisfy the required # of frames.
        start_ts = std::uniform_int_distribution<>(ts1, ts2)(meta_randgen);
        // seek a frame at start_ts
        const int64_t seekingTo = 0 > (start_ts - margin) ? \
          0 : (start_ts - margin);
        ret = av_seek_frame(
            inputContext,
            hasVideo ? videoStreamIndex_ : audioStreamIndex_,
            seekingTo,
            AVSEEK_FLAG_BACKWARD);
        VLOG(2) << "Seeking to "
                << av_rescale_q(seekingTo, stream->time_base, AV_TIME_BASE_Q)
                << " / " << duration;
        // if we need to decode from the start_frm
      } else if (params.decode_type_ == DecodeType::USE_START_FRM) {
        if (videoStream_ == nullptr) {
          LOG(ERROR) << ("Nullptr found at videoStream_");
        }
        start_ts = int(floor(
            (videoStream_->duration * start_frm)
                / (videoStream_->nb_frames)));
        // seek a frame at start_ts
        ret = av_seek_frame(
            inputContext,
            hasVideo ? videoStreamIndex_ : audioStreamIndex_,
            0 > (start_ts - margin) ? 0 : (start_ts - margin),
            AVSEEK_FLAG_BACKWARD);
      } else {
        mustDecodeAll = true;
      }

      if (ret < 0) {
        LOG(INFO) << "Unable to decode from a random start point";
        /* fall back to default decoding of all frames from start */
        av_seek_frame(inputContext,
            hasVideo ? videoStreamIndex_ : audioStreamIndex_,
            0,
            AVSEEK_FLAG_BACKWARD);
        mustDecodeAll = true;
      }
    } else {
      mustDecodeAll = true;
    }

    int gotPicture = 0;
    int eof = 0;
    int selectiveDecodedFrames = 0;

    int maxFrames = (params.decode_type_ == DecodeType::DO_UNIFORM_SMP)
                    ? MAX_DECODING_FRAMES
                    : params.num_of_required_frame_;
    // There is a delay between reading packets from the
    // transport and getting decoded frames back.
    // Therefore, after EOF, continue going while
    // the decoder is still giving us frames.
    int ipacket = 0;
    bool audioDecodeNeeded = hasAudio;
    bool videoDecodeNeeded = hasVideo;
    while (audioDecodeNeeded || videoDecodeNeeded){
      audioDecodeNeeded = hasAudio && !eof;
      videoDecodeNeeded = hasVideo &&
          ((!eof || gotPicture) &&
          /* either you must decode all frames or decode upto maxFrames
           * based on status of the mustDecodeAll flag */
          (mustDecodeAll ||
              ((!mustDecodeAll) && (selectiveDecodedFrames < maxFrames))) &&
          /* If on the last interval and not autodecoding keyframes and a
           * SpecialFps indicates no more frames are needed, stop decoding */
          !((itvlIter == params.intervals_.end() &&
              (currFps == SpecialFps::SAMPLE_TIMESTAMP_ONLY ||
                  currFps == SpecialFps::SAMPLE_NO_FRAME)) &&
              !params.keyFrames_));
      try {
        if (!eof) {
          ret = av_read_frame(inputContext, &packet);
          if (ret == AVERROR_EOF) {
            eof = 1;
            av_free_packet(&packet);
            packet.data = NULL;
            packet.size = 0;
            // stay in the while loop to flush frames
          } else if (ret == AVERROR(EAGAIN)) {
            av_free_packet(&packet);
            continue;
          } else if (ret < 0) {
            LOG(ERROR) << "Error reading packet : " << ffmpegErrorStr(ret);
          }
          ipacket++;

          auto si = packet.stream_index;
          if (params.getAudio_ && audioStreamIndex_ >= 0 &&
              si == audioStreamIndex_ && audioDecodeNeeded) {
            // Audio packets can have multiple audio frames in a single packet
            while (packet.size > 0) {
              if (audioCodecContext_ == nullptr ||
                audioStreamFrame_ == nullptr ||
                convertCtx_ == nullptr) {
                continue;
              }
              getAudioSample(
                  packet,
                  audioCodecContext_,
                  audioStreamFrame_,
                  convertCtx_,
                  callback,
                  params);
            }
            if (audioStreamFrame_ == nullptr) {
              continue;
            }
            av_frame_unref(audioStreamFrame_);
          }

          if (si != videoStreamIndex_) {
            av_free_packet(&packet);
            continue;
          }
        }

        if (params.getVideo_ && videoStreamIndex_ >= 0 && videoDecodeNeeded) {
          if (videoCodecContext_ == nullptr or videoStreamFrame_ == nullptr) {
            continue;
          }
          ret = avcodec_decode_video2(
              videoCodecContext_, videoStreamFrame_, &gotPicture, &packet);
          if (ret < 0) {
            LOG(ERROR) << "Error decoding video frame : " << ffmpegErrorStr(ret);
          }
          try {
            // Nothing to do without a picture
            if (!gotPicture) {
              av_free_packet(&packet);
              continue;
            }
            frameIndex++;

            if (videoStreamFrame_ == nullptr) {
              continue;
            }
            long int frame_ts =
                av_frame_get_best_effort_timestamp(videoStreamFrame_);
            timestamp = frame_ts * av_q2d(videoStream_->time_base);
            if ((frame_ts >= start_ts && !mustDecodeAll) || mustDecodeAll) {
              /* process current frame if:
               * 1) We are not doing selective decoding and mustDecodeAll
               *    OR
               * 2) We are doing selective decoding and current frame
               *   timestamp is >= start_ts from where we start selective
               *   decoding*/
              // if reaching the next interval, update the current fps
              // and reset lastFrameTimestamp so the current frame could be
              // sampled (unless fps == SpecialFps::SAMPLE_NO_FRAME)
              if (itvlIter != params.intervals_.end() &&
                  timestamp >= itvlIter->timestamp) {
                lastFrameTimestamp = -1.0;
                currFps = itvlIter->fps;
                prevTimestamp = itvlIter->timestamp;
                itvlIter++;
                if (itvlIter != params.intervals_.end() &&
                    prevTimestamp >= itvlIter->timestamp) {
                  LOG(ERROR) << (
                      "Sampling interval timestamps must be strictly "
                          "ascending.");
                }
              }

              // keyFrame will bypass all checks on fps sampling settings
              bool keyFrame = params.keyFrames_ && videoStreamFrame_->key_frame;
              if (!keyFrame) {
                //if fps == SpecialFps::SAMPLE_NO_FRAME (0), don't sample at all
                if (currFps == SpecialFps::SAMPLE_NO_FRAME) {
                  av_free_packet(&packet);
                  continue;
                }

                // fps is considered reached in the following cases:
                // 1. lastFrameTimestamp < 0 - start of a new interval
                //    (or first frame)
                // 2. currFps == SpecialFps::SAMPLE_ALL_FRAMES (-1) - sample
                //    every frame
                // 3. timestamp - lastFrameTimestamp has reached target fps and
                //    currFps > 0 (not special fps setting)
                // different modes for fps:
                // SpecialFps::SAMPLE_NO_FRAMES (0):
                //     disable fps sampling, no frame sampled at all
                // SpecialFps::SAMPLE_ALL_FRAMES (-1):
                //     unlimited fps sampling, will sample at native video fps
                // SpecialFps::SAMPLE_TIMESTAMP_ONLY (-2):
                //     disable fps sampling, but will get the frame at specific
                //     timestamp
                // others (> 0): decoding at the specified fps
                bool fpsReached = lastFrameTimestamp < 0 ||
                    currFps == SpecialFps::SAMPLE_ALL_FRAMES ||
                    (currFps > 0 &&
                        timestamp >= lastFrameTimestamp + (1 / currFps));

                if (!fpsReached) {
                  av_free_packet(&packet);
                  continue;
                }
              }

              lastFrameTimestamp = timestamp;

              outputFrameIndex++;
              if (params.maximumOutputFrames_ != -1 &&
                  outputFrameIndex >= params.maximumOutputFrames_) {
                // enough frames
                av_free_packet(&packet);
                break;
              }

              AVFrame* rgbFrame = av_frame_alloc();
              if (!rgbFrame) {
                LOG(ERROR) << ("Error allocating AVframe");
              }

              try {
                // Determine required buffer size and allocate buffer
                int numBytes =
                    avpicture_get_size(pixFormat, outWidth, outHeight);
                DecodedFrame::AvDataPtr buffer(
                    (uint8_t*) av_malloc(numBytes * sizeof(uint8_t)));

                int size = avpicture_fill(
                    (AVPicture*) rgbFrame,
                    buffer.get(),
                    pixFormat,
                    outWidth,
                    outHeight);
                if (scaleContext_ == nullptr) {
                  continue;
                }
                sws_scale(
                    scaleContext_,
                    videoStreamFrame_->data,
                    videoStreamFrame_->linesize,
                    0,
                    videoCodecContext_->height,
                    rgbFrame->data,
                    rgbFrame->linesize);

                unique_ptr<DecodedFrame> frame = make_unique<DecodedFrame>();
                frame->width_ = outWidth;
                frame->height_ = outHeight;
                frame->data_ = move(buffer);
                frame->size_ = size;
                frame->index_ = frameIndex;
                frame->outputFrameIndex_ = outputFrameIndex;
                frame->timestamp_ = timestamp;
                frame->keyFrame_ = videoStreamFrame_->key_frame;
                callback.frameDecoded(std::move(frame));

                selectiveDecodedFrames++;
                av_frame_free(&rgbFrame);
              } catch (const std::exception&) {
                av_frame_free(&rgbFrame);
              }
            }
            if (videoStreamFrame_ != nullptr) {
              av_frame_unref(videoStreamFrame_);
            }
          } catch (const std::exception&) {
            if (videoStreamFrame_ != nullptr) {
              av_frame_unref(videoStreamFrame_);
            }
          }
        }
        av_free_packet(&packet);
      } catch (const std::exception& exception) {
        LOG(ERROR) << "Caught an exception" << exception.what();
        av_free_packet(&packet);
      }
    } // of while loop
    callback.videoDecodingEnded(timestamp);

    // free all stuffs
    if (scaleContext_ != nullptr) {
      sws_freeContext(scaleContext_);
    }
    swr_free(&convertCtx_);
    av_packet_unref(&packet);
    av_frame_free(&videoStreamFrame_);
    av_frame_free(&audioStreamFrame_);
    if (videoCodecContext_ != nullptr) {
      avcodec_close(videoCodecContext_);
    }
    if (audioCodecContext_ != nullptr) {
      avcodec_close(audioCodecContext_);
    }
    avformat_close_input(&inputContext);
    avformat_free_context(inputContext);
  } catch (const std::exception& exception) {
    LOG(ERROR) << "Caught an exception" << exception.what();
    // In case of decoding error
    // free all stuffs
    sws_freeContext(scaleContext_);
    swr_free(&convertCtx_);
    av_packet_unref(&packet);
    av_frame_free(&videoStreamFrame_);
    av_frame_free(&audioStreamFrame_);
    avcodec_close(videoCodecContext_);
    avcodec_close(audioCodecContext_);
    avformat_close_input(&inputContext);
    avformat_free_context(inputContext);
  }
}

void AVDecoder::decodeMemory(
    const string& videoName,
    const char* buffer,
    const int size,
    const Params& params,
    const int start_frm,
    Callback& callback,
    int& number_of_frames) {
  VideoIOContext ioctx(buffer, size);
  decodeLoop(videoName, ioctx, params, start_frm, callback, number_of_frames);
}

void AVDecoder::decodeFile(
    const string& file,
    const Params& params,
    const int start_frm,
    Callback& callback,
    int& number_of_frames) {
  VideoIOContext ioctx(file);
  decodeLoop(file, ioctx, params, start_frm, callback, number_of_frames);
}

string AVDecoder::ffmpegErrorStr(int result) {
  std::array<char, 128> buf;
  av_strerror(result, buf.data(), buf.size());
  return string(buf.data());
}

void FreeAVDecodedData(
    std::vector<std::unique_ptr<DecodedFrame>>& sampledFrames,
    std::vector<float>& audioSamples
) {

  // free the sampledFrames and sampledAudio
  for (int i = 0; i < sampledFrames.size(); i++) {
    DecodedFrame* p = sampledFrames[i].release();
    delete p;
  }

  sampledFrames.clear();
  audioSamples.clear();
}

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
) {
  std::vector<std::unique_ptr<DecodedFrame>> sampledFrames;
  std::vector<float> sampledAudio;
  AVDecoder decoder;

  CallbackImpl callback;
  // decoding from buffer or file
  if (!use_local_file) {
    decoder.decodeMemory(string("Memory Buffer"), video_buffer,
        encoded_size, params, start_frm, callback, number_of_frames);
  } else {
    decoder.decodeFile(
      video_filename, params, start_frm, callback, number_of_frames);
  }
  for (auto& frame : callback.frames) {
    sampledFrames.push_back(move(frame));
  }
  int n=0;
  for (auto sample : callback.audio_samples) {
    buffer_audio.push_back(sample);
    n++;
  }

  for (int i = 0; i < buffer_rgb.size(); i++) {
    unsigned char* buff = buffer_rgb[i];
    delete []buff;
  }
  buffer_rgb.clear();

  if (params.getVideo_) {
    if (sampledFrames.size() < params.num_of_required_frame_) {
      // LOG(ERROR) << "The video seems faulty and we could not decode enough "
      //  << " frames: sampledFrames.size() << " VS "
      //  << params.num_of_required_frame_;
      FreeAVDecodedData(sampledFrames, sampledAudio);
      return true;
    }
    if (sampledFrames.size() == 0) {
      LOG(ERROR) << "The samples frames have size 0, no frame to process";
      FreeAVDecodedData(sampledFrames, sampledAudio);
      return true;
    }
    height = sampledFrames[0]->height_;
    width = sampledFrames[0]->width_;
    clip_start_frame = sampledFrames[0]->index_;
    float
        sample_stepsz = (clip_per_video <= 1) ? 0 : (float(sampledFrames.size()
        - params.num_of_required_frame_) / (clip_per_video - 1));

    int image_size = 3 * height * width;
    int clip_size = params.num_of_required_frame_ * image_size;
    // get the RGB frames for each clip
    for (int i = 0; i < clip_per_video; i++) {
      unsigned char* buffer_rgb_ptr = new unsigned char[clip_size];
      int clip_start = floor(i * sample_stepsz);
      for (int j = 0; j < params.num_of_required_frame_; j++) {
        memcpy(buffer_rgb_ptr + j * image_size,
            (unsigned char*) sampledFrames[j + clip_start]->data_.get(),
            image_size * sizeof(unsigned char));
      }
      buffer_rgb.push_back(buffer_rgb_ptr);
    }
  }
  FreeAVDecodedData(sampledFrames, sampledAudio);

  return true;
}

} // namespace video_modeling

} // namespace caffe2
