from torchvision.datasets import Kinetics400


class Kinetics(Kinetics400):
    def __init__(
        self,
        root,
        frames_per_clip,
        step_between_clips=1,
        frame_rate=None,
        extensions=("avi",),
        transform=None,
        _precomputed_metadata=None,
        num_workers=1,
        _video_width=0,
        _video_height=0,
        _video_min_dimension=0,
        _audio_samples=0,
        _audio_channels=0,
    ):

        super(Kinetics, self).__init__(
            root,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            extensions,
            transform,
            _precomputed_metadata,
            num_workers,
            _video_width,
            _video_height,
            _video_min_dimension,
            _audio_samples,
            _audio_channels,
        )

    def __getitem__(self, idx):
        video, _, info, video_idx = self.video_clips.get_clip(idx)
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label, video_idx, clip_idx
