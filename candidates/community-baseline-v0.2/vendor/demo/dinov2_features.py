import torch
from safetensors.torch import load_file as load_safe_tensors
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import pickle
import gzip
from pathlib import Path
from video_reader import VideoReader
from device import selected_device
DINOV2_SOURCE = Path(os.environ.get('SHUBERT_DINOV2_SOURCE', Path(__file__).resolve().parents[1] / 'dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8'))
import argparse
import json
import csv
import glob
import time
from typing import List, Union, Optional, Tuple

class DINOEmbedder:
    """
    A class for extracting DINOv2 embeddings from video frames or images.
    """

    def __init__(self, dino_model_path: str, batch_size: int=128, device: Optional[str]=None):
        """
        Initialize the DINOEmbedder.
        
        Args:
            dino_model_path: Path to the fine-tuned DINOv2 model
            batch_size: Batch size for processing frames
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None
        """
        self.dino_model_path = dino_model_path
        self.batch_size = batch_size
        self.device = torch.device(device) if device else selected_device()
        self.model = self._load_dino_model()
        self.model.eval()
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        print(f'DINOEmbedder initialized on device: {self.device}')

    def _load_dino_model(self) -> nn.Module:
        """Load the fine-tuned DINOv2 model."""
        if not (DINOV2_SOURCE / 'hubconf.py').is_file():
            raise FileNotFoundError(f'pinned local DINOv2 source missing: {DINOV2_SOURCE}')
        model = torch.hub.load(str(DINOV2_SOURCE), 'dinov2_vits14_reg', source='local', pretrained=False)
        pretrained = {'teacher': load_safe_tensors(self.dino_model_path, device='cpu')}
        new_state_dict = {}
        for key, value in pretrained['teacher'].items():
            if 'dino_head' in key:
                continue
            else:
                new_key = key.replace('backbone.', '')
                new_state_dict[new_key] = value
        pos_embed = nn.Parameter(torch.zeros(1, 257, 384))
        model.pos_embed = pos_embed
        model.load_state_dict(new_state_dict, strict=True)
        model.to(self.device)
        return model

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame."""
        if isinstance(frame, np.ndarray):
            image = Image.fromarray(frame)
        else:
            image = frame
        tensor = self.transform(image)
        return tensor[:3]

    def _preprocess_frames_batch(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess a batch of frames."""
        batch_tensors = torch.stack([self._preprocess_frame(frame) for frame in frames])
        return batch_tensors.to(self.device)

    def extract_embeddings_from_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract DINOv2 embeddings from a list of frames.
        
        Args:
            frames: List of frames as numpy arrays
            
        Returns:
            Numpy array of embeddings with shape (num_frames, embedding_dim)
        """
        all_embeddings = []
        for idx in range(0, len(frames), self.batch_size):
            batch_frames = frames[idx:idx + self.batch_size]
            batch_tensors = self._preprocess_frames_batch(batch_frames)
            with torch.no_grad():
                batch_embeddings = self.model(batch_tensors).cpu().numpy()
            all_embeddings.append(batch_embeddings)
        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings

    def extract_embeddings_from_video(self, video_input: Union[str, VideoReader], target_size: Tuple[int, int]=(224, 224)) -> np.ndarray:
        """
        Extract DINOv2 embeddings from a video.
        
        Args:
            video_input: Either a path to video file (str) or a VideoReader object
            target_size: Target size for video frames (width, height)
            
        Returns:
            Numpy array of embeddings with shape (num_frames, embedding_dim)
        """
        if isinstance(video_input, str):
            video_path = Path(video_input)
            if not video_path.exists():
                raise FileNotFoundError(f'Video file not found: {video_input}')
            try:
                vr = VideoReader(str(video_path), width=target_size[0], height=target_size[1])
            except Exception as e:
                raise RuntimeError(f'Error loading video {video_input}: {e}')
        else:
            vr = video_input
        total_frames = len(vr)
        all_embeddings = []
        for idx in range(0, total_frames, self.batch_size):
            batch_indices = range(idx, min(idx + self.batch_size, total_frames))
            batch_frames = vr[batch_indices]
            batch_tensors = self._preprocess_frames_batch(batch_frames)
            with torch.no_grad():
                batch_embeddings = self.model(batch_tensors).cpu().numpy()
            all_embeddings.append(batch_embeddings)
        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings

    def extract_embeddings_from_video_and_save(self, video_path: str, output_folder: str) -> str:
        """
        Extract embeddings from video and save to file.
        
        Args:
            video_path: Path to the video file
            output_folder: Folder to save the embeddings
            
        Returns:
            Path to the saved embeddings file
        """
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        embeddings = self.extract_embeddings_from_video(video_path)
        video_name = Path(video_path).stem
        np_path = Path(output_folder) / f'{video_name}.npy'
        np.save(np_path, embeddings)
        return str(np_path)

    def extract_embedding_from_single_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract DINOv2 embedding from a single image.
        
        Args:
            image: Image as numpy array or PIL Image
            
        Returns:
            Numpy array of embedding with shape (1, embedding_dim)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor).cpu().numpy()
        return embedding

def extract_embeddings_from_frames(frames: List[np.ndarray], dino_model_path: str, batch_size: int=128) -> np.ndarray:
    """
    Convenience function to extract embeddings from frames.
    
    Args:
        frames: List of frames as numpy arrays
        dino_model_path: Path to the fine-tuned DINOv2 model
        batch_size: Batch size for processing
        
    Returns:
        Numpy array of embeddings
    """
    embedder = DINOEmbedder(dino_model_path, batch_size)
    return embedder.extract_embeddings_from_frames(frames)

def extract_embeddings_from_video(video_path: str, dino_model_path: str, batch_size: int=128) -> np.ndarray:
    """
    Convenience function to extract embeddings from video.
    
    Args:
        video_path: Path to the video file
        dino_model_path: Path to the fine-tuned DINOv2 model
        batch_size: Batch size for processing
        
    Returns:
        Numpy array of embeddings
    """
    embedder = DINOEmbedder(dino_model_path, batch_size)
    return embedder.extract_embeddings_from_video(video_path)

def video_to_embeddings(video_path: str, output_folder: str, dino_path: str, batch_size: int=128):
    """
    Original function for backward compatibility with command-line usage.
    """
    try:
        embedder = DINOEmbedder(dino_path, batch_size)
        embedder.extract_embeddings_from_video_and_save(video_path, output_folder)
    except Exception as e:
        print(f'Error processing {video_path}: {e}')

def get_mp4_files(directory: str) -> List[str]:
    """Get all MP4 files in a directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f'Directory not found: {directory}')
    mp4_files = glob.glob(os.path.join(directory, '*.mp4'))
    return [os.path.abspath(file) for file in mp4_files]

def load_file(filename: str):
    """Load a pickled and gzipped file."""
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

def is_string_in_file(file_path: str, target_string: str) -> bool:
    """Check if a string exists in a file."""
    try:
        with Path(file_path).open('r') as f:
            for line in f:
                if target_string in line:
                    return True
        return False
    except Exception as e:
        print(f'Error: {e}')
        return False

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True, help='index of the sub_list to work with')
    parser.add_argument('--time_limit', type=int, required=True, help='time limit in seconds')
    parser.add_argument('--batch_size', type=int, required=True, help='number of videos to process in this batch')
    parser.add_argument('--files_list', type=str, required=True, help='path to the files list file')
    parser.add_argument('--output_folder', type=str, required=True, help='path to the output folder')
    parser.add_argument('--dino_path', type=str, required=True, help='path to the dino model')
    args = parser.parse_args()
    start_time = time.time()
    fixed_list = load_file(args.files_list)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    embedder = DINOEmbedder(args.dino_path, batch_size=512)
    video_batches = [fixed_list[i:i + args.batch_size] for i in range(0, len(fixed_list), args.batch_size)]
    print(f'Total number of video batches: {len(video_batches)}')
    for video_path in video_batches[args.index]:
        current_time = time.time()
        if current_time - start_time > args.time_limit:
            print('Time limit reached. Stopping execution.')
            break
        video_name = Path(video_path).stem
        np_path = Path(args.output_folder) / f'{video_name}.npy'
        if np_path.exists():
            print(f'Skipping {video_path} - output already exists')
            continue
        else:
            try:
                print(f'Processing {video_path}')
                embedder.extract_embeddings_from_video_and_save(video_path, args.output_folder)
                print(f'Successfully processed {video_path}')
            except Exception as e:
                print(f'Error processing {video_path}: {e}')
