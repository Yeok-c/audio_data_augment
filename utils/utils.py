import os
import numpy as np
import tensorflow as tf
from pathlib import Path

class utils():
    def __init__(self):
        pass
    
    # Get the list of all noise files, with specific endings
    def get_noise_paths(self, dataset_noise_path, ending=".wav"):
        noise_paths = []
        for subdir in os.listdir(dataset_noise_path):
            subdir_path = Path(dataset_noise_path) / subdir
            if os.path.isdir(subdir_path):
                noise_paths += [
                    os.path.join(subdir_path, filepath)
                    for filepath in os.listdir(subdir_path)
                    if filepath.endswith(ending)
                ]

        print(
            "Found {} files belonging to {} directories".format(
                len(noise_paths), len(os.listdir(dataset_noise_path))
            )
        )
        return noise_paths

    # Split noise into chunks of 16000 each
    def load_noise_sample(self, path, sampling_rate):
        sample, sr = tf.audio.decode_wav(
            tf.io.read_file(path), desired_channels=1
        )
        if sr == sampling_rate:
            # Number of slices of 16000 each that can be generated from the noise sample
            slices = int(sample.shape[0] / sampling_rate)
            sample = tf.split(sample[: slices * sampling_rate], slices)
            print("{} slices, {} samples generated from {}".format(slices, sample, path))
            return sample
        else:
            print("Sampling rate of {} for {} is incorrect. Ignoring it".format(sr, path))
            return None
            
    def path_to_audio(self, path, sampling_rate):
        """Reads and decodes an audio file."""
        audio = tf.io.read_file(path)
        audio, _ = tf.audio.decode_wav(audio, 1, sampling_rate)
        return audio


    def paths_and_labels_to_dataset(self, audio_paths, labels, sampling_rate):
        """Constructs a dataset of audios and labels."""
        path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
        audio_ds = path_ds.map(lambda x: self.path_to_audio(x, sampling_rate))
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        return tf.data.Dataset.zip((audio_ds, label_ds))

    def add_noise(self, audio, noises=None, scale=0.5):
        if noises is not None:
            # Create a random tensor of the same size as audio ranging from
            # 0 to the number of noise stream samples that we have.
            tf_rnd = tf.random.uniform(
                (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
            )
            noise = tf.gather(noises, tf_rnd, axis=0)

            # Get the amplitude proportion between the audio and the noise
            prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
            prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

            # Adding the rescaled noise to audio
            audio = audio + noise * prop * scale

        return audio