import os
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from collections.abc import Callable

class VideoFrameDataset(torch.utils.data.Dataset):

    def __init__(self,
                 root_path: str,
                 annotationfile_path: str,
                 seq_len: int,
                 imagefile_template: str='{:05d}.jpg',
                 transform = None):
        super(VideoFrameDataset, self).__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.seq_len = seq_len

        self._parse_list()

    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert('RGB')]

    def _parse_list(self):
        self.frame_list = []
        for fname in os.listdir(self.root_path):
            item = (os.path.join(root_path, fname), )
            frame_list.append()
        self.frame_list = [VideoRecord(x.strip().split(' '), self.root_path) for x in open(self.annotationfile_path)]


    def _sample_indices(self, record):
        """
        For each segment, chooses an index from where frames
        are to be loaded from.
        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            List of indices of where the frames of each
            segment are to be loaded from.
        """

        average_duration = (record.num_frames - self.frames_per_segment + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + np.random.randint(average_duration, size=self.num_segments)

        # edge cases for when a video only has a tiny number of frames.
        elif record.num_frames > self.num_segments:
            offsets = np.sort(np.random.randint(record.num_frames - self.frames_per_segment + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def __getitem__(self, index):
        """
        For video with id index, loads self.NUM_SEGMENTS * self.FRAMES_PER_SEGMENT
        frames from evenly chosen locations.
        Args:
            index: Video sample index.
        Returns:
            a list of PIL images or the result
            of applying self.transform on this list if
            self.transform is not None.
        """
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self._get(record, segment_indices)

    def _get(self, record, indices):
        """
        Loads the frames of a video at the corresponding
        indices.
        Args:
            record: VideoRecord denoting a video sample.
            indices: Indices at which to load video frames from.
        Returns:
            1) A list of PIL images or the result
            of applying self.transform on this list if
            self.transform is not None.
            2) An integer denoting the video label.
        """

        images = list()
        for seg_ind in indices:
            frame_index = int(seg_ind)
            for i in range(self.frames_per_segment):
                seg_img = self._load_image(record.path, frame_index)
                images.extend(seg_img)
                if frame_index < record.num_frames:
                    frame_index += 1

        if self.transform is not None:
            images = self.transform(images)

        return images, record.label

    def __len__(self):
        return len(self.video_list)

class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``VideoFrameDataset``.
    """
    def forward(self, img_list):
        """
        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.
        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])