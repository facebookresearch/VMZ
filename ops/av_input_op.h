#ifndef AV_INPUT_OP_H_
#define AV_INPUT_OP_H_

#include <istream>
#include <ostream>
#include <random>
#include <string>

#include "caffe2/core/db.h"
#include "caffe2/core/logging.h"
#include "caffe2/operators/prefetch_op.h"
#include "caffe2/utils/math.h"
#include "c10/core/thread_pool.h"
#include "av_decoder.h"
#include "av_io.h"

namespace caffe2 {

namespace video_modeling {

const int kNumLogMelFilters = 40;
const int kNumLogMelFrames = 100;
const int kWindowLength = 16;
const int kWindowStep = 10;
const int kAudioSamplingRate = 16000;

template <class Context>
class AVInputOp final : public PrefetchOperator<Context> {
 public:
  using OperatorBase::OutputSize;
  using PrefetchOperator<Context>::context_;
  using PrefetchOperator<Context>::prefetch_thread_;
  explicit AVInputOp(const OperatorDef& operator_def, Workspace* ws);
  ~AVInputOp() {
    PrefetchOperator<Context>::Finalize();
  }

  // override methods
  bool Prefetch() override;
  bool CopyPrefetched() override;

 private:
  void CheckParamsAndPrint();

  bool GetClipsAndLabelsFromDBValue(
      const std::string& value,
      int& height,
      int& width,
      std::vector<unsigned char*>& buffer_rgb,
      std::vector<float>& buffer_logmels,
      int* label_data,
      int64_t* video_id_data,
      int& number_of_frames,
      int& clip_start_frame,
      std::mt19937* randgen
  );

  void DecodeAndTransform(
      const std::string& value,
      float* clip_rgb_data,
      float* clip_of_logmels_data,
      int* label_data,
      int64_t* video_id_data,
      std::mt19937* randgen,
      std::bernoulli_distribution* mirror_this_clip
  );

  const db::DBReader* reader_;
  CPUContext cpu_context_;
  Tensor prefetched_clip_rgb_{CPU};
  Tensor prefetched_clip_of_{CPU};
  Tensor prefetched_clip_logmels_{CPU};
  Tensor prefetched_label_{CPU};
  Tensor prefetched_video_id_{CPU};
  Tensor prefetched_clip_rgb_on_device_{Context::GetDeviceType()};
  Tensor prefetched_clip_of_on_device_{Context::GetDeviceType()};
  Tensor prefetched_clip_logmels_on_device_{Context::GetDeviceType()};
  Tensor prefetched_label_on_device_{Context::GetDeviceType()};
  Tensor prefetched_video_id_on_device_{Context::GetDeviceType()};
  int batch_size_;
  int clip_per_video_;
  std::vector<float> mean_rgb_;
  std::vector<float> inv_std_rgb_;
  std::vector<float> mean_of_;
  std::vector<float> inv_std_of_;
  int channels_rgb_;
  int channels_of_;
  int crop_size_;
  int scale_h_;
  int scale_w_;
  int short_edge_;
  std::vector<int> jitter_scales_;
  int length_rgb_;
  int sampling_rate_rgb_;
  int num_of_required_frame_;
  int length_of_;
  int sampling_rate_of_;
  int frame_gap_of_;
  bool random_mirror_;
  int num_of_class_;
  bool use_local_file_;
  bool random_crop_;
  int decode_type_;
  int video_res_type_;
  bool get_rgb_;
  bool get_logmels_;
  bool get_video_id_;
  bool do_multi_label_;
  int logMelFrames_;
  int logMelFilters_;
  int logMelWindowSizeMs_;
  int logMelWindowStepMs_;
  int logMelAudioSamplingRate_;

  // 0 for no alignment
  // 1 for perfect align
  // 2 align by first
  // 3 align by center
  int align_audio_;
  // if 0, we don't interpolate; otherwise we sample the audio length and
  // interpolate to logMelFrames_
  int audio_length_;
  bool tune_audio_step_;

  // thread pool for parse + decode
  int num_decode_threads_;
  std::shared_ptr<TaskThreadPool> thread_pool_;
};

template <class Context>
void AVInputOp<Context>::CheckParamsAndPrint() {

  // check whether the input parameters are valid or not
  CAFFE_ENFORCE_GT(batch_size_, 0, "Batch size should be positive.");
  if (get_rgb_) {
    CAFFE_ENFORCE_GT(
        clip_per_video_, 0, "Number of clips per video should be positive.");
    CAFFE_ENFORCE_GT(crop_size_, 0, "Must provide the cropping value.");
    CAFFE_ENFORCE_GT(
        num_of_required_frame_, 0, "Required number of frames must be positive."
    );

    if (video_res_type_ == VideoResType::USE_SHORT_EDGE) {
      CAFFE_ENFORCE_GT(short_edge_, 0, "Must provide the short edge value.");
      CAFFE_ENFORCE_GE(
          short_edge_,
          crop_size_,
          "The short edge must be no smaller than the crop value.");
    } else if (video_res_type_ == VideoResType::USE_WIDTH_HEIGHT) {
      CAFFE_ENFORCE_GT(scale_h_, 0, "Must provide the scale height value.");
      CAFFE_ENFORCE_GT(scale_w_, 0, "Must provide the scale width value.");
      CAFFE_ENFORCE_GE(
          scale_h_,
          crop_size_,
          "The scaled height must be no smaller than the crop value.");
      CAFFE_ENFORCE_GE(
          scale_w_,
          crop_size_,
          "The scaled width must be no smaller than the crop value.");
    }
  }

  if (jitter_scales_.size() > 0) {
    CAFFE_ENFORCE_GE(
      video_res_type_,
      VideoResType::USE_SHORT_EDGE,
      "Scale jittering is used with short_edge scaling only"
    );
  }

  if (get_rgb_) {
    CAFFE_ENFORCE_GT(length_rgb_, 0, "Must provide rgb clip length.");
    CAFFE_ENFORCE_GT(
        sampling_rate_rgb_, 0, "4 frames for mc2; 2 frames for res3d.");
    CAFFE_ENFORCE_EQ(
        channels_rgb_, mean_rgb_.size(), "Number rgb channels is wrong!");
    CAFFE_ENFORCE_EQ(
        channels_rgb_, inv_std_rgb_.size(), "Number rgb channels is wrong!");
  }

  if (clip_per_video_ > 1) {
    CAFFE_ENFORCE_EQ(
        decode_type_,
        DecodeType::DO_UNIFORM_SMP,
        "Only uniformly sampling is supported when sampling multiple clips!");
  }

  if (do_multi_label_) {
    CAFFE_ENFORCE_GT(
        num_of_class_,
        0,
        "Number of classes must be set when using multiple labels.");
  }

  if (get_logmels_) {
    CAFFE_ENFORCE_GT(
        logMelFrames_,
        0,
        "Number of log mel frames must be set when using log mel feature.");
    CAFFE_ENFORCE_GT(
        logMelFilters_,
        0,
        "Number of log mel filters must be set when using log mel feature.");
    CAFFE_ENFORCE_GT(
        logMelWindowSizeMs_,
        0,
        "Audio Window size must be set when using log mel feature.");
    CAFFE_ENFORCE_GT(
        logMelWindowStepMs_,
        0,
        "Audio Window step must be set when using log mel feature.");
    CAFFE_ENFORCE_GT(
        logMelAudioSamplingRate_,
        0,
        "Audio sampling rate must be set when using log mel feature.");
  }

  // print out the parameter settings
  LOG(INFO) << "Creating a clip input op with the following setting: ";
  LOG(INFO) << "    Using " << num_decode_threads_ << " CPU threads;";
  LOG(INFO) << "    Outputting in batches of " << batch_size_ << " videos;";
  LOG(INFO) << "    Each video has " << clip_per_video_ << " clips;";
  LOG(INFO) << "    Scaling image to " << scale_h_ << "x" << scale_w_;
  LOG(INFO) << "    Cropping video frame to " << crop_size_
            << (random_mirror_ ? " with " : " without ") << "random mirroring;";
  LOG(INFO) << "    Using " << (random_crop_ ? "random" : "center") << " crop";

  if (get_rgb_) {
    LOG(INFO) << "    Using a clip of " << length_rgb_ << " rgb frames "
              << "with " << channels_rgb_ << " channels "
              << "and a sampling rate of 1:" << sampling_rate_rgb_;
    for (int i = 0; i < channels_rgb_; i++) {
      LOG(INFO) << "    RGB " << i << "-th channel mean: " << mean_rgb_[i]
                << " std: " << 1.f / inv_std_rgb_[i];
    }
  }

  if (get_logmels_) {
    LOG(INFO) << "    Using log mels with"
              << logMelFilters_ << " filters "
              << logMelFrames_ << " frames "
              << logMelWindowSizeMs_ << " window size (ms) "
              << logMelWindowStepMs_ << " window step (ms) "
              << logMelAudioSamplingRate_ << " audio sampling rate ";
  }

  if (video_res_type_ == VideoResType::ORIGINAL_RES) {
    LOG(INFO) << "    Use original resolution";
  } else if (video_res_type_ == VideoResType::USE_SHORT_EDGE) {
    LOG(INFO) << "    Resize and keep aspect ratio";
  } else if (video_res_type_ == VideoResType::USE_WIDTH_HEIGHT) {
    LOG(INFO) << "    Resize and ignore aspect ratio";
  } else {
    LOG(ERROR) << "    Unknown video resolution type";
  }

  if (video_res_type_ == VideoResType::USE_SHORT_EDGE) {
    if (jitter_scales_.size() > 0) {
      LOG(INFO) << "Using scale jittering:";
      for (int idx = 0; idx < jitter_scales_.size(); idx++) {
        LOG(INFO) << "scale " << idx <<": "<< jitter_scales_[idx];
      }
    } else {
      LOG(INFO) << "No scale jittering is used.";
    }
  }

  if (decode_type_ == DecodeType::DO_TMP_JITTER) {
    LOG(INFO) << "    Do temporal jittering";
  } else if (decode_type_ == DecodeType::USE_START_FRM) {
    LOG(INFO) << "    Use start_frm for decoding";
  } else if (decode_type_ == DecodeType::DO_UNIFORM_SMP) {
    LOG(INFO) << "    Do uniformly sampling";
  } else {
    LOG(ERROR) << "    Unknown video decoding type";
  }
}

template <class Context>
AVInputOp<Context>::AVInputOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : PrefetchOperator<Context>(operator_def, ws),
      reader_(nullptr),
      batch_size_(
          OperatorBase::template GetSingleArgument<int>("batch_size", 0)),
      clip_per_video_(
          OperatorBase::template GetSingleArgument<int>("clip_per_video", 1)),
      channels_rgb_(
          OperatorBase::template GetSingleArgument<int>("channels_rgb", 3)),
      channels_of_(
          OperatorBase::template GetSingleArgument<int>("channels_of", 2)),
      crop_size_(OperatorBase::template GetSingleArgument<int>("crop_size", 0)),
      scale_h_(OperatorBase::template GetSingleArgument<int>("scale_h", 0)),
      scale_w_(OperatorBase::template GetSingleArgument<int>("scale_w", 0)),
      short_edge_(
          OperatorBase::template GetSingleArgument<int>("short_edge", 0)),
      jitter_scales_(OperatorBase::template GetRepeatedArgument<int>(
          "jitter_scales",
          {})),
      length_rgb_(
          OperatorBase::template GetSingleArgument<int>("length_rgb", 0)),
      sampling_rate_rgb_(OperatorBase::template GetSingleArgument<int>(
          "sampling_rate_rgb",
          1)),
      length_of_(OperatorBase::template GetSingleArgument<int>("length_of", 0)),
      sampling_rate_of_(
          OperatorBase::template GetSingleArgument<int>("sampling_rate_of", 1)),
      frame_gap_of_(
          OperatorBase::template GetSingleArgument<int>("frame_gap_of", 1)),
      random_mirror_(OperatorBase::template GetSingleArgument<bool>(
          "random_mirror",
          true)),
      num_of_class_(
          OperatorBase::template GetSingleArgument<int>("num_of_class", 0)),
      use_local_file_(OperatorBase::template GetSingleArgument<bool>(
          "use_local_file",
          false)),
      random_crop_(
          OperatorBase::template GetSingleArgument<bool>("random_crop", true)),
      decode_type_(
          OperatorBase::template GetSingleArgument<int>("decode_type", 0)),
      video_res_type_(
          OperatorBase::template GetSingleArgument<int>("video_res_type", 0)),
      get_rgb_(OperatorBase::template GetSingleArgument<bool>(
          "get_rgb",
          false)),
      get_logmels_(OperatorBase::template GetSingleArgument<bool>("get_logmels",
          false)),
      get_video_id_(OperatorBase::template GetSingleArgument<bool>(
          "get_video_id",
          false)),
      do_multi_label_(OperatorBase::template GetSingleArgument<bool>(
          "do_multi_label",
          false)),
      logMelFrames_(OperatorBase::template GetSingleArgument<int>(
          "logmel_frames", kNumLogMelFrames)),
      logMelFilters_(OperatorBase::template GetSingleArgument<int>(
          "logmel_filters", kNumLogMelFilters)),
      logMelWindowSizeMs_(OperatorBase::template GetSingleArgument<int>(
          "logmel_winsize_ms", kWindowLength)),
      logMelWindowStepMs_(OperatorBase::template GetSingleArgument<int>(
          "logmel_winstep_ms", kWindowStep)),
      logMelAudioSamplingRate_(OperatorBase::template GetSingleArgument<int>(
          "logmel_audio_sr", kAudioSamplingRate)),
      align_audio_(OperatorBase::template GetSingleArgument<int>(
          "align_audio", 1)),
      audio_length_(OperatorBase::template GetSingleArgument<int>(
          "audio_length", 0)),
      tune_audio_step_(OperatorBase::template GetSingleArgument<bool>(
          "tune_audio_step",
          false)),
      num_decode_threads_(OperatorBase::template GetSingleArgument<int>(
          "num_decode_threads", 4)),
      thread_pool_(std::make_shared<TaskThreadPool>(num_decode_threads_)) {
  try {
    num_of_required_frame_ = 0;
    // mean and std for normalizing different optical flow data type;
    // Note that the statistics are generated from SOA, and you may
    // want to change them if you are running on a different dataset;
    // Each dimension represents: horizontal component of optical flow,
    // vertical component of optical flow, magnitude of optical flow,
    // Gray, R, G, B.
    const std::vector<float> InputDataMean = {0.0046635, 0.0046261,
        0.963986, 102.976, 110.201, 100.64, 95.9966};
    const std::vector<float> InputDataStd = {0.972347, 0.755146,
        1.43588, 55.3691, 58.1489, 56.4701, 55.3324};
    // if we need RGB as an input
    if (get_rgb_) {
      // how many frames we need for RGB
      num_of_required_frame_ = std::max(
          num_of_required_frame_, (length_rgb_ - 1) * sampling_rate_rgb_ + 1);

      channels_rgb_ = 3;
      for (int i = 4; i < 7; i++) {
        mean_rgb_.push_back(InputDataMean[i]);
        inv_std_rgb_.push_back(1.f / InputDataStd[i]);
      }
    }

    CheckParamsAndPrint();
    // Always need a dbreader, even when using local video files
    CAFFE_ENFORCE_GT(
        operator_def.input_size(), 0, "Need to have a DBReader blob input");

    vector<int64_t> data_shape(5);
    vector<int64_t> label_shape(2);
    vector<int64_t> logmels_shape(4);

    // for RGB data
    data_shape[0] = batch_size_ * clip_per_video_;
    data_shape[1] = channels_rgb_;
    data_shape[2] = length_rgb_;
    data_shape[3] = crop_size_;
    data_shape[4] = crop_size_;
    prefetched_clip_rgb_.Resize(data_shape);

    // for audio data
    logmels_shape[0] = batch_size_ * clip_per_video_;
    logmels_shape[1] = 1;
    logmels_shape[2] = logMelFrames_;
    logmels_shape[3] = logMelFilters_;
    prefetched_clip_logmels_.Resize(logmels_shape);

    // If do_multi_label is used, output label is a binary vector
    // of length num_of_class indicating which labels present
    if (do_multi_label_) {
      label_shape[0] = batch_size_ * clip_per_video_;
      label_shape[1] = num_of_class_;
      prefetched_label_.Resize(label_shape);
    } else {
      prefetched_label_.Resize(
          vector<int64_t>(1, batch_size_ * clip_per_video_));
    }

    prefetched_video_id_.Resize(
        vector<int64_t>(1, batch_size_ * clip_per_video_));
  } catch (const std::exception& exc) {
    std::cerr << "While calling AVInputOp initialization\n";
    std::cerr << exc.what();
  }
}

template <class Context>
bool AVInputOp<Context>::GetClipsAndLabelsFromDBValue(
    const std::string& value,
    int& height,
    int& width,
    std::vector<unsigned char*>& buffer_rgb,
    std::vector<float>& buffer_logmels,
    int* label_data,
    int64_t* video_id_data,
    int& number_of_frames,
    int& clip_start_frame,
    std::mt19937* randgen
) {
  try {
    TensorProtos protos;
    int curr_proto_idx = 0;
    CAFFE_ENFORCE(protos.ParseFromString(value));
    const TensorProto& video_proto = protos.protos(curr_proto_idx++);
    const TensorProto& label_proto = protos.protos(curr_proto_idx++);

    int start_frm = 0;
    // start_frm is only valid when sampling 1 clip per video without
    // temporal jitterring
    if (decode_type_ == DecodeType::USE_START_FRM) {
      CAFFE_ENFORCE_GE(
          protos.protos_size(),
          curr_proto_idx + 1,
          "Start frm proto not provided");
      const TensorProto& start_frm_proto = protos.protos(curr_proto_idx++);
      start_frm = start_frm_proto.int32_data(0);
    }

    if (get_video_id_) {
      CAFFE_ENFORCE_GE(
          protos.protos_size(), curr_proto_idx + 1, "Video Id not provided");
      const TensorProto& video_id_proto = protos.protos(curr_proto_idx);
      for (int i = 0; i < clip_per_video_; i++) {
        video_id_data[i] = video_id_proto.int64_data(0);
      }
    }

    // assign labels
    if (!do_multi_label_) {
      for (int i = 0; i < clip_per_video_; i++) {
        label_data[i] = label_proto.int32_data(0);
      }
    } else {
      // For multiple label case, output label is a binary vector
      // where presented concepts are makred 1
      memset(label_data, 0, sizeof(int) * num_of_class_ * clip_per_video_);
      for (int i = 0; i < clip_per_video_; i++) {
        for (int j = 0; j < label_proto.int32_data_size(); j++) {
          CAFFE_ENFORCE_LT(
              label_proto.int32_data(j),
              num_of_class_,
              "Label should be less than the number of classes.");
          label_data[i * num_of_class_ + label_proto.int32_data(j)] = 1;
        }
      }
    }

    if (use_local_file_) {
      CAFFE_ENFORCE_EQ(
          video_proto.data_type(),
          TensorProto::STRING,
          "Database with a file_list is expected to be string data");
    }

    // initializing the decoding params
    Params params;
    params.maximumOutputFrames_ = MAX_DECODING_FRAMES;
    params.video_res_type_ = video_res_type_;
    params.crop_size_ = crop_size_;
    params.short_edge_ = short_edge_;
    params.outputWidth_ = scale_w_;
    params.outputHeight_ = scale_h_;
    params.decode_type_ = decode_type_;
    params.num_of_required_frame_ = num_of_required_frame_;
    params.getAudio_ = get_logmels_;
    params.getVideo_ = get_rgb_;
    params.outrate_ = logMelAudioSamplingRate_;

    if (jitter_scales_.size() > 0) {
      int select_idx =
        std::uniform_int_distribution<>(0, jitter_scales_.size() - 1)(*randgen);
      params.short_edge_ = jitter_scales_[select_idx];
    }

    char* video_buffer = nullptr; // for decoding from buffer
    std::string video_filename; // for decoding from file
    int encoded_size = 0;
    if (video_proto.data_type() == TensorProto::STRING) {
      const string& encoded_video_str = video_proto.string_data(0);
      if (!use_local_file_) {
        encoded_size = encoded_video_str.size();
        video_buffer = const_cast<char*>(encoded_video_str.data());
      } else {
        video_filename = encoded_video_str;
      }
    } else if (video_proto.data_type() == TensorProto::BYTE) {
      if (!use_local_file_) {
        encoded_size = video_proto.byte_data().size();
        video_buffer = const_cast<char*>(video_proto.byte_data().data());
      } else {
        // TODO: does this works?
        video_filename = video_proto.string_data(0);
      }
    } else {
      CAFFE_ENFORCE(false, "Unknown video data type.");
    }

    DecodeMultipleAVClipsFromVideo(
        video_buffer,
        video_filename,
        encoded_size,
        params,
        start_frm,
        clip_per_video_,
        use_local_file_,
        height,
        width,
        buffer_rgb,
        buffer_logmels,
        number_of_frames,
        clip_start_frame
    );
  } catch (const std::exception& exc) {
    std::cerr << "While calling GetClipsAndLabelsFromDBValue()\n";
    std::cerr << exc.what();
  }
  return true;
}

template <class Context>
void AVInputOp<Context>::DecodeAndTransform(
    const std::string& value,
    float* clip_rgb_data,
    float* clip_of_logmels_data,
    int* label_data,
    int64_t* video_id_data,
    std::mt19937* randgen,
    std::bernoulli_distribution* mirror_this_clip) {
  try {
    std::vector<unsigned char*> buffer_rgb;
    // get the video resolution after decoding
    int height = 0;
    int width = 0;
    // get the number of visual frames to use for synchronizing with aduio
    int number_of_frames = 0;
    int clip_start_frame = 0;
    // Decode the video from memory or read from a local file
    std::vector<float> audioSamples;
    audioSamples.reserve(logMelAudioSamplingRate_);
    CHECK(GetClipsAndLabelsFromDBValue(
        value, height, width, buffer_rgb, audioSamples, label_data,
        video_id_data, number_of_frames, clip_start_frame, randgen));

    int clip_offset_rgb = channels_rgb_ * length_rgb_ * crop_size_ * crop_size_;
    int clip_offset_of = channels_of_ * length_of_ * crop_size_ * crop_size_;
    for (int i = 0; i < std::min(clip_per_video_, int(buffer_rgb.size()));
         i++) {
      // get the rectangle for cropping
      int h_off = 0;
      int w_off = 0;
      if (random_crop_) {
        // using random crop for training
        h_off =
            std::uniform_int_distribution<>(0, height - crop_size_)(*randgen);
        w_off =
            std::uniform_int_distribution<>(0, width - crop_size_)(*randgen);
      } else {
        // using center crop for testing
        h_off = (height - crop_size_) / 2;
        w_off = (width - crop_size_) / 2;
      }
      cv::Rect rect(w_off, h_off, crop_size_, crop_size_);

      // randomly mirror the image or not
      bool mirror_me = random_mirror_ && (*mirror_this_clip)(*randgen);

      if (get_rgb_ && clip_rgb_data) {
        ClipTransformRGB(
            buffer_rgb[i],
            crop_size_,
            length_rgb_,
            channels_rgb_,
            sampling_rate_rgb_,
            height,
            width,
            h_off,
            w_off,
            mirror_me,
            mean_rgb_,
            inv_std_rgb_,
            clip_rgb_data + (i * clip_offset_rgb));
      }
    }

    if (get_logmels_ && clip_of_logmels_data) {
      ClipTransformAudioLogmel(
          decode_type_,
          get_rgb_,
          clip_rgb_data,
          number_of_frames,
          tune_audio_step_,
          logMelFrames_,
          logMelAudioSamplingRate_,
          logMelWindowSizeMs_,
          logMelWindowStepMs_,
          logMelFilters_,
          num_of_required_frame_,
          align_audio_,
          clip_per_video_,
          audio_length_,
          clip_start_frame,
          audioSamples,
          clip_of_logmels_data);
    }

    if (buffer_rgb.size() > 0) {
      for (int i = 0; i < buffer_rgb.size(); i++) {
        unsigned char* buff = buffer_rgb[i];
        delete[] buff;
      }
    }
    buffer_rgb.clear();
  } catch (const std::exception& exc) {
    std::cerr << "While calling DecodeAndTransform()\n";
    std::cerr << exc.what();
  }
}

template <class Context>
bool AVInputOp<Context>::Prefetch() {
  try {
    // We will get the reader pointer from input.
    // If we use local clips, db will store the list
    reader_ = &OperatorBase::Input<db::DBReader>(0);

    // Call mutable_data() once to allocate the underlying memory.
    prefetched_clip_rgb_.mutable_data<float>();
    prefetched_clip_of_.mutable_data<float>();
    prefetched_clip_logmels_.mutable_data<float>();
    prefetched_label_.mutable_data<int>();
    prefetched_video_id_.mutable_data<int64_t>();

    // Prefetching handled with a thread pool of "decode_threads" threads.
    std::mt19937 meta_randgen(time(nullptr));
    std::vector<std::mt19937> randgen_per_thread;
    for (int i = 0; i < num_decode_threads_; ++i) {
      randgen_per_thread.emplace_back(meta_randgen());
    }

    std::bernoulli_distribution mirror_this_clip(0.5);
    for (int item_id = 0; item_id < batch_size_; ++item_id) {
      std::mt19937* randgen =
          &randgen_per_thread[item_id % num_decode_threads_];

      int frame_size = crop_size_ * crop_size_;
      // get the clip data pointer for the item_id -th example
      float* clip_rgb_data = prefetched_clip_rgb_.mutable_data<float>() +
          frame_size * length_rgb_ * channels_rgb_ * item_id * clip_per_video_;

      // get the logmels data for the current clip
      float* clip_logmels_data =
          prefetched_clip_logmels_.mutable_data<float>() +
              item_id * logMelFilters_ * logMelFrames_ * clip_per_video_;

      // get the label data pointer for the item_id -th example
      int* label_data = prefetched_label_.mutable_data<int>() +
          (do_multi_label_ ? num_of_class_ : 1) * item_id * clip_per_video_;

      // get the video id data pointer for the item_id -th example
      int64_t* video_id_data =
          prefetched_video_id_.mutable_data<int64_t>() + item_id * clip_per_video_;

      std::string key, value;
      // read data
      reader_->Read(&key, &value);

      thread_pool_->run(std::bind(
          &AVInputOp<Context>::DecodeAndTransform,
          this,
          std::string(value),
          clip_rgb_data,
          clip_logmels_data,
          label_data,
          video_id_data,
          randgen,
          &mirror_this_clip));
    } // for over the batch
    thread_pool_->waitWorkComplete();

    // If the context is not CPUContext, we will need to do a copy in the
    // prefetch function as well.
    if (!std::is_same<Context, CPUContext>::value) {
      if (get_rgb_) {
        prefetched_clip_rgb_on_device_.CopyFrom(
            prefetched_clip_rgb_, &context_);
      }
      if (get_logmels_) {
        prefetched_clip_logmels_on_device_.CopyFrom(
            prefetched_clip_logmels_, &context_);
      }
      prefetched_label_on_device_.CopyFrom(prefetched_label_, &context_);
      if (get_video_id_) {
        prefetched_video_id_on_device_.CopyFrom(
            prefetched_video_id_, &context_);
      }
    }
  } catch (const std::exception& exc) {
    std::cerr << "While calling Prefetch()\n";
    std::cerr << exc.what();
  }
  return true;
}

template <class Context>
bool AVInputOp<Context>::CopyPrefetched() {
  try {
    int index = 0;
    auto type = Context::GetDeviceType();
    if (get_rgb_) {
      auto* clip_rgb_output = OperatorBase::Output<Tensor>(index++, type);
      if (std::is_same<Context, CPUContext>::value) {
        clip_rgb_output->CopyFrom(prefetched_clip_rgb_, &context_);
      } else {
        clip_rgb_output->CopyFrom(prefetched_clip_rgb_on_device_, &context_);
      }
    }

    if (get_logmels_) {
      auto* clip_logmels_output =
          OperatorBase::Output<Tensor>(index++, type);
      if (std::is_same<Context, CPUContext>::value) {
        clip_logmels_output->CopyFrom(prefetched_clip_logmels_, &context_);
      } else {
        clip_logmels_output->CopyFrom(
            prefetched_clip_logmels_on_device_, &context_);
      }
    }

    auto* label_output = OperatorBase::Output<Tensor>(index++, type);
    if (std::is_same<Context, CPUContext>::value) {
      label_output->CopyFrom(prefetched_label_, &context_);
    } else {
      label_output->CopyFrom(prefetched_label_on_device_, &context_);
    }

    if (get_video_id_) {
      auto* video_id_output = OperatorBase::Output<Tensor>(index, type);
      if (std::is_same<Context, CPUContext>::value) {
        video_id_output->CopyFrom(prefetched_video_id_, &context_);
      } else {
        video_id_output->CopyFrom(prefetched_video_id_on_device_, &context_);
      }
    }

  } catch (const std::exception& exc) {
    std::cerr << "While calling CopyPrefetched()\n";
    std::cerr << exc.what();
  }

  return true;
}

} // namespace video_modeling

} // namespace caffe2

#endif // AV_INPUT_OP_H_
