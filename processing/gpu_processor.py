"""
GPU-Accelerated Processing Pipeline
Separate processing frame for fast medical image analysis
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessingStatus(str, Enum):
    """Processing status enum"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class ProcessingTask:
    """Represents a GPU processing task"""
    task_id: str
    status: ProcessingStatus = ProcessingStatus.IDLE
    progress: float = 0.0
    result: Optional[Dict] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0


class GPUProcessingPipeline:
    """
    GPU-accelerated processing pipeline
    Manages background processing tasks with GPU queue
    """

    def __init__(self, max_concurrent: int = 2, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_concurrent = max_concurrent
        self.is_gpu = str(self.device) == "cuda"
        
        # Task queue
        self.task_queue: Dict[str, ProcessingTask] = {}
        self.active_tasks: List[str] = []
        
        # GPU memory management
        self.enable_memory_optimization = True
        
        if self.is_gpu:
            logger.info(f"✅ GPU Pipeline Ready on {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠️  Processing pipeline using CPU")

    async def process_segmentation_async(
        self,
        task_id: str,
        image: np.ndarray,
        model,
    ) -> Dict:
        """
        Async GPU segmentation processing
        
        Args:
            task_id: Unique task identifier
            image: Input medical image
            model: Segmentation model
            
        Returns:
            Processing result with segmentation mask
        """
        task = ProcessingTask(task_id=task_id)
        task.start_time = datetime.now()
        self.task_queue[task_id] = task
        
        try:
            # Wait for GPU slot
            await self._wait_for_gpu_slot()
            
            task.status = ProcessingStatus.PROCESSING
            task.progress = 0.1
            
            # Move to GPU
            img_tensor = torch.from_numpy(image).float().to(self.device)
            task.progress = 0.2
            
            # Normalize
            img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
            task.progress = 0.3
            
            # Run model inference
            with torch.no_grad():
                output = model(img_tensor.unsqueeze(0).unsqueeze(0))
                task.progress = 0.7
            
            # Post-process
            mask = torch.softmax(output, dim=1).argmax(dim=1).squeeze()
            mask_np = mask.cpu().numpy()
            
            task.progress = 0.9
            
            # Clear GPU memory
            self._clear_gpu_memory()
            
            task.status = ProcessingStatus.COMPLETED
            task.progress = 1.0
            task.result = {
                "mask": mask_np,
                "num_classes": output.shape[1],
                "shape": mask_np.shape,
            }
            
        except Exception as e:
            logger.error(f"Segmentation processing failed: {e}")
            task.status = ProcessingStatus.ERROR
            task.error = str(e)
        
        finally:
            task.end_time = datetime.now()
            task.duration_ms = (task.end_time - task.start_time).total_seconds() * 1000
            self.active_tasks.remove(task_id) if task_id in self.active_tasks else None
        
        return task

    async def process_depth_async(
        self,
        task_id: str,
        image: np.ndarray,
        model,
    ) -> Dict:
        """
        Async GPU depth estimation
        
        Args:
            task_id: Unique task identifier
            image: Input medical image
            model: Depth estimation model
            
        Returns:
            Processing result with depth map
        """
        task = ProcessingTask(task_id=task_id)
        task.start_time = datetime.now()
        self.task_queue[task_id] = task
        
        try:
            await self._wait_for_gpu_slot()
            
            task.status = ProcessingStatus.PROCESSING
            task.progress = 0.1
            
            # Convert to tensor
            if len(image.shape) == 2:
                img_tensor = torch.from_numpy(image).unsqueeze(0).float().to(self.device)
            else:
                img_tensor = torch.from_numpy(image).float().to(self.device)
            
            task.progress = 0.2
            
            # Normalize
            img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
            
            # Run inference
            with torch.no_grad():
                depth = model(img_tensor.unsqueeze(0))
                task.progress = 0.7
            
            # Process output
            depth_np = depth.squeeze().cpu().numpy()
            
            # Invert if needed (closer = higher value)
            depth_np = 1.0 / (depth_np + 1e-8)
            
            task.progress = 0.9
            self._clear_gpu_memory()
            
            task.status = ProcessingStatus.COMPLETED
            task.progress = 1.0
            task.result = {
                "depth_map": depth_np,
                "shape": depth_np.shape,
                "min_depth": float(depth_np.min()),
                "max_depth": float(depth_np.max()),
            }
            
        except Exception as e:
            logger.error(f"Depth processing failed: {e}")
            task.status = ProcessingStatus.ERROR
            task.error = str(e)
        
        finally:
            task.end_time = datetime.now()
            task.duration_ms = (task.end_time - task.start_time).total_seconds() * 1000
            self.active_tasks.remove(task_id) if task_id in self.active_tasks else None
        
        return task

    async def process_super_resolution_async(
        self,
        task_id: str,
        image: np.ndarray,
        model,
        scale_factor: int = 4,
    ) -> Dict:
        """
        Async GPU super-resolution upscaling
        
        Args:
            task_id: Unique task identifier
            image: Input image
            model: SR model
            scale_factor: Upscaling factor
            
        Returns:
            Processing result with upscaled image
        """
        task = ProcessingTask(task_id=task_id)
        task.start_time = datetime.now()
        self.task_queue[task_id] = task
        
        try:
            await self._wait_for_gpu_slot()
            
            task.status = ProcessingStatus.PROCESSING
            task.progress = 0.1
            
            # Prepare tensor
            img_tensor = torch.from_numpy(image).float().to(self.device)
            if img_tensor.dim() == 2:
                img_tensor = img_tensor.unsqueeze(0)
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            task.progress = 0.2
            
            # Normalize
            img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
            
            # Run SR model
            with torch.no_grad():
                sr_image = model(img_tensor)
                task.progress = 0.7
            
            # Post-process
            sr_np = sr_image.squeeze().cpu().numpy()
            sr_np = np.clip(sr_np, 0, 1)
            
            task.progress = 0.9
            self._clear_gpu_memory()
            
            task.status = ProcessingStatus.COMPLETED
            task.progress = 1.0
            task.result = {
                "image": sr_np,
                "original_shape": image.shape,
                "upscaled_shape": sr_np.shape,
                "scale_factor": scale_factor,
            }
            
        except Exception as e:
            logger.error(f"Super-resolution processing failed: {e}")
            task.status = ProcessingStatus.ERROR
            task.error = str(e)
        
        finally:
            task.end_time = datetime.now()
            task.duration_ms = (task.end_time - task.start_time).total_seconds() * 1000
            self.active_tasks.remove(task_id) if task_id in self.active_tasks else None
        
        return task

    async def process_volumetric_async(
        self,
        task_id: str,
        volume: np.ndarray,
        model,
    ) -> Dict:
        """
        Async GPU volumetric 3D processing
        
        Args:
            task_id: Unique task identifier
            volume: 3D volume data
            model: 3D CNN model
            
        Returns:
            Processing result with features
        """
        task = ProcessingTask(task_id=task_id)
        task.start_time = datetime.now()
        self.task_queue[task_id] = task
        
        try:
            await self._wait_for_gpu_slot()
            
            task.status = ProcessingStatus.PROCESSING
            task.progress = 0.1
            
            # Prepare volume tensor
            vol_tensor = torch.from_numpy(volume).float().to(self.device)
            if vol_tensor.dim() == 3:
                vol_tensor = vol_tensor.unsqueeze(0).unsqueeze(0)
            
            task.progress = 0.2
            
            # Normalize
            vol_tensor = (vol_tensor - vol_tensor.min()) / (vol_tensor.max() - vol_tensor.min() + 1e-8)
            
            # Chunk processing for large volumes
            chunk_size = 32
            if vol_tensor.shape[2] > chunk_size:
                task.result = await self._process_volume_chunks(vol_tensor, model, task)
            else:
                with torch.no_grad():
                    features = model(vol_tensor)
                    task.progress = 0.7
                
                task.result = {
                    "features": features.cpu().numpy(),
                    "shape": features.shape,
                }
            
            task.progress = 0.9
            self._clear_gpu_memory()
            
            task.status = ProcessingStatus.COMPLETED
            task.progress = 1.0
            
        except Exception as e:
            logger.error(f"Volumetric processing failed: {e}")
            task.status = ProcessingStatus.ERROR
            task.error = str(e)
        
        finally:
            task.end_time = datetime.now()
            task.duration_ms = (task.end_time - task.start_time).total_seconds() * 1000
            self.active_tasks.remove(task_id) if task_id in self.active_tasks else None
        
        return task

    async def _process_volume_chunks(
        self,
        volume: torch.Tensor,
        model,
        task: ProcessingTask,
    ) -> Dict:
        """Process large volumes in chunks"""
        chunk_size = 32
        all_features = []
        
        num_chunks = (volume.shape[2] + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, volume.shape[2])
            
            chunk = volume[:, :, start_idx:end_idx, :, :]
            
            with torch.no_grad():
                features = model(chunk)
                all_features.append(features.cpu())
            
            # Update progress
            task.progress = 0.2 + (i / num_chunks) * 0.5
            await asyncio.sleep(0)  # Yield control
        
        # Concatenate results
        combined = torch.cat(all_features, dim=2)
        
        return {
            "features": combined.numpy(),
            "shape": combined.shape,
            "num_chunks": num_chunks,
        }

    async def process_batch_async(
        self,
        task_id: str,
        images: List[np.ndarray],
        model,
        process_type: str = "segmentation",
    ) -> Dict:
        """
        Process multiple images in batch
        
        Args:
            task_id: Unique task identifier
            images: List of images
            model: Processing model
            process_type: Type of processing
            
        Returns:
            Batch processing results
        """
        task = ProcessingTask(task_id=task_id)
        task.start_time = datetime.now()
        self.task_queue[task_id] = task
        
        try:
            await self._wait_for_gpu_slot()
            
            task.status = ProcessingStatus.PROCESSING
            
            # Stack images
            batch = np.stack(images)
            batch_tensor = torch.from_numpy(batch).float().to(self.device)
            
            task.progress = 0.2
            
            # Normalize
            batch_tensor = (batch_tensor - batch_tensor.min()) / (batch_tensor.max() - batch_tensor.min() + 1e-8)
            
            # Process all at once on GPU
            with torch.no_grad():
                outputs = model(batch_tensor)
                task.progress = 0.7
            
            results_np = outputs.cpu().numpy()
            
            task.progress = 0.9
            self._clear_gpu_memory()
            
            task.status = ProcessingStatus.COMPLETED
            task.progress = 1.0
            task.result = {
                "outputs": results_np,
                "batch_size": len(images),
                "shape": results_np.shape,
            }
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            task.status = ProcessingStatus.ERROR
            task.error = str(e)
        
        finally:
            task.end_time = datetime.now()
            task.duration_ms = (task.end_time - task.start_time).total_seconds() * 1000
            self.active_tasks.remove(task_id) if task_id in self.active_tasks else None
        
        return task

    def get_task_status(self, task_id: str) -> Optional[ProcessingTask]:
        """Get current task status"""
        return self.task_queue.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a processing task"""
        if task_id in self.task_queue:
            task = self.task_queue[task_id]
            if task.status == ProcessingStatus.PROCESSING:
                task.status = ProcessingStatus.CANCELLED
                if task_id in self.active_tasks:
                    self.active_tasks.remove(task_id)
                return True
        return False

    async def _wait_for_gpu_slot(self):
        """Wait until GPU slot is available"""
        while len(self.active_tasks) >= self.max_concurrent:
            await asyncio.sleep(0.1)
        
        # Find a new task ID
        new_task_id = len([t for t in self.task_queue])
        self.active_tasks.append(str(new_task_id))

    def _clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if self.is_gpu and self.enable_memory_optimization:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_gpu_stats(self) -> Dict:
        """Get GPU statistics"""
        if not self.is_gpu:
            return {"device": "CPU", "status": "disabled"}
        
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return {
            "device": torch.cuda.get_device_name(0),
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "utilization_percent": (allocated / total) * 100,
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
        }


# Singleton instance
_pipeline_instance = None


def get_gpu_pipeline(max_concurrent: int = 2) -> GPUProcessingPipeline:
    """Get or create GPU processing pipeline singleton"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = GPUProcessingPipeline(max_concurrent=max_concurrent)
    return _pipeline_instance
