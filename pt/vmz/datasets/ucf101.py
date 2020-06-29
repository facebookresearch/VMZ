from torchvision.datasets.ucf101 import UCF101


class UCF(UCF101):
    def __init__(
        self,
        root,
        annotation_path,
        frames_per_clip,
        step_between_clips=1,
        frame_rate=None,
        fold=1,
        train=True,
        transform=None,
        _precomputed_metadata=None,
        num_workers=1,
        _video_width=0,
        _video_height=0,
        _video_min_dimension=0,
        _audio_samples=0,
    ):

        super(UCF, self).__init__(
            root,
            annotation_path,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            fold,
            train,
            transform,
            _precomputed_metadata,
            num_workers,
            _video_width,
            _video_height,
            _video_min_dimension,
            _audio_samples,
        )

    def __getitem__(self, idx):
        video, _, info, video_idx = self.video_clips.get_clip(idx)
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)

        label = self.samples[self.indices[video_idx]][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label, video_idx, clip_idx
