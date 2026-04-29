"""
Fallback PyTorch-like module for when torch is unavailable.
Provides minimal dummy implementations for model definitions.
"""

import logging

logger = logging.getLogger(__name__)

class Module:
    """Dummy base class for model modules."""
    def __init__(self, *args, **kwargs):
        """Accept any arguments (no-op)."""
        pass
    
    def __call__(self, *args, **kwargs):
        """Support calling as a function - return dummy tensor output."""
        # When called as a model, return a dummy tensor
        # Extract input tensor to infer output shape
        import numpy as np
        if len(args) > 0:
            input_tensor = args[0]
            if hasattr(input_tensor, 'data'):
                # Return dummy tensor with same shape as input
                dummy_output = np.random.rand(*input_tensor.data.shape).astype(np.float32)
                return Tensor(dummy_output)
        # Fallback: return dummy tensor
        return Tensor(np.random.rand(1, 1, 64, 64).astype(np.float32))
    
    def to(self, device):
        """Move module to device (no-op for fallback)."""
        return self
    
    def eval(self):
        """Set module to eval mode."""
        return self
    
    def train(self):
        """Set module to train mode."""
        return self
    
    def parameters(self):
        """Return empty list of parameters."""
        return []
    
    def state_dict(self):
        """Return empty state dict."""
        return {}
    
    def load_state_dict(self, state_dict):
        """No-op state dict load."""
        return self

class ModuleList(list):
    """Dummy modular list container."""
    def __init__(self, modules=None):
        super().__init__(modules or [])

class ParameterList(list):
    """Dummy parameter list."""
    def __init__(self, params=None):
        super().__init__(params or [])

class Conv2d(Module):
    """Dummy Conv2d layer."""
    def __init__(self, *args, **kwargs):
        pass

class ConvTranspose2d(Module):
    """Dummy transpose convolution."""
    def __init__(self, *args, **kwargs):
        pass

class BatchNorm2d(Module):
    """Dummy batch norm."""
    def __init__(self, *args, **kwargs):
        pass

class ReLU(Module):
    """Dummy ReLU."""
    def __init__(self, *args, **kwargs):
        pass

class Linear(Module):
    """Dummy linear layer."""
    def __init__(self, *args, **kwargs):
        pass

class Dropout(Module):
    """Dummy dropout."""
    def __init__(self, *args, **kwargs):
        pass

class GELU(Module):
    """Dummy GELU."""
    def __init__(self, *args, **kwargs):
        pass

class Sequential(Module):
    """Dummy sequential container."""
    def __init__(self, *args, **kwargs):
        self.layers = args

class AvgPool2d(Module):
    """Dummy average pooling."""
    def __init__(self, *args, **kwargs):
        pass

class MaxPool2d(Module):
    """Dummy max pooling."""
    def __init__(self, *args, **kwargs):
        pass

class LayerNorm(Module):
    """Dummy layer normalization."""
    def __init__(self, *args, **kwargs):
        pass

class Embedding(Module):
    """Dummy embedding layer."""
    def __init__(self, *args, **kwargs):
        pass

class MultiheadAttention(Module):
    """Dummy multihead attention."""
    def __init__(self, *args, **kwargs):
        pass

class TransformerEncoder(Module):
    """Dummy transformer encoder."""
    def __init__(self, *args, **kwargs):
        pass

class TransformerEncoderLayer(Module):
    """Dummy transformer encoder layer."""
    def __init__(self, *args, **kwargs):
        pass

class LeakyReLU(Module):
    """Dummy leaky ReLU."""
    def __init__(self, *args, **kwargs):
        pass

class PReLU(Module):
    """Dummy parametric ReLU."""
    def __init__(self, *args, **kwargs):
        pass

class Sigmoid(Module):
    """Dummy sigmoid."""
    def __init__(self, *args, **kwargs):
        pass

class Tanh(Module):
    """Dummy tanh."""
    def __init__(self, *args, **kwargs):
        pass

class AdaptiveAvgPool2d(Module):
    """Dummy adaptive average pooling."""
    def __init__(self, *args, **kwargs):
        pass

class InstanceNorm2d(Module):
    """Dummy instance norm."""
    def __init__(self, *args, **kwargs):
        pass

class GroupNorm(Module):
    """Dummy group norm."""
    def __init__(self, *args, **kwargs):
        pass

class Pad(Module):
    """Dummy padding."""
    def __init__(self, *args, **kwargs):
        pass

class ReflectionPad2d(Module):
    """Dummy reflection padding."""
    def __init__(self, *args, **kwargs):
        pass

class AdaptiveMaxPool2d(Module):
    """Dummy adaptive max pooling."""
    def __init__(self, *args, **kwargs):
        pass

class Conv3d(Module):
    """Dummy 3D convolution."""
    def __init__(self, *args, **kwargs):
        pass

class BatchNorm3d(Module):
    """Dummy 3D batch norm."""
    def __init__(self, *args, **kwargs):
        pass

class MaxPool3d(Module):
    """Dummy 3D max pooling."""
    def __init__(self, *args, **kwargs):
        pass

class AdaptiveAvgPool3d(Module):
    """Dummy 3D adaptive average pooling."""
    def __init__(self, *args, **kwargs):
        pass

class Flatten(Module):
    """Dummy flatten layer."""
    def __init__(self, *args, **kwargs):
        pass

# Create module-like objects with dynamic attribute handling
class NNModule:
    """Neural network module namespace with dynamic fallback for unknown layers."""
    Module = Module
    ModuleList = ModuleList
    ParameterList = ParameterList
    Conv2d = Conv2d
    ConvTranspose2d = ConvTranspose2d
    BatchNorm2d = BatchNorm2d
    ReLU = ReLU
    LeakyReLU = LeakyReLU
    PReLU = PReLU
    Sigmoid = Sigmoid
    Tanh = Tanh
    Linear = Linear
    Dropout = Dropout
    GELU = GELU
    Sequential = Sequential
    AvgPool2d = AvgPool2d
    MaxPool2d = MaxPool2d
    AdaptiveAvgPool2d = AdaptiveAvgPool2d
    AdaptiveMaxPool2d = AdaptiveMaxPool2d
    LayerNorm = LayerNorm
    InstanceNorm2d = InstanceNorm2d
    GroupNorm = GroupNorm
    Embedding = Embedding
    MultiheadAttention = MultiheadAttention
    TransformerEncoder = TransformerEncoder
    TransformerEncoderLayer = TransformerEncoderLayer
    Pad = Pad
    ReflectionPad2d = ReflectionPad2d
    Conv3d = Conv3d
    BatchNorm3d = BatchNorm3d
    MaxPool3d = MaxPool3d
    AdaptiveAvgPool3d = AdaptiveAvgPool3d
    Flatten = Flatten
    
    def __getattribute__(self, name):
        """Return a dummy module class for any unknown attribute."""
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if not name.startswith('_'):
                return Module
            raise

nn = NNModule()

class F:
    """Dummy functional module."""
    @staticmethod
    def relu(*args, **kwargs):
        return None
    
    @staticmethod
    def interpolate(input_tensor, size=None, scale_factor=None, mode='nearest', **kwargs):
        """Interpolate tensor to new size."""
        import numpy as np
        if hasattr(input_tensor, 'data') and input_tensor.data is not None:
            data = input_tensor.data
            if size is not None:
                # Resize using numpy (simplified interpolation)
                return Tensor(data)  # Return as-is for dummy implementation
            elif scale_factor is not None:
                return Tensor(data)  # Return as-is for dummy implementation
        return input_tensor
    
    @staticmethod
    def avg_pool2d(*args, **kwargs):
        return None
    
    @staticmethod
    def max_pool2d(*args, **kwargs):
        return None

class Tensor:
    """Dummy tensor."""
    def __init__(self, data=None):
        self.data = data
        self.shape = getattr(data, 'shape', None)
    
    def unsqueeze(self, dim):
        """Add dimension to tensor."""
        if self.data is not None:
            import numpy as np
            self.data = np.expand_dims(self.data, axis=dim)
        return self
    
    def squeeze(self, dim=None):
        """Remove dimensions."""
        if self.data is not None:
            import numpy as np
            if dim is None:
                self.data = np.squeeze(self.data)
            else:
                self.data = np.squeeze(self.data, axis=dim)
        return self
    
    def float(self):
        """Convert to float type."""
        if self.data is not None:
            self.data = self.data.astype('float32')
        return self
    
    def to(self, device):
        """Move tensor to device (no-op)."""
        return self
    
    def cpu(self):
        """Move to CPU (no-op)."""
        return self
    
    def numpy(self):
        """Convert to numpy."""
        import numpy as np
        if isinstance(self.data, np.ndarray):
            return self.data
        return np.array(self.data) if self.data is not None else np.array([])
    
    def __getattr__(self, name):
        """Fallback for unknown attributes."""
        return lambda *args, **kwargs: self

def tensor(data, *args, **kwargs):
    """Dummy tensor constructor."""
    return Tensor(data)

class NoGradContext:
    """Dummy no_grad context manager."""
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass

class torch:
    """Dummy torch module."""
    Module = Module
    nn = nn
    F = F
    Tensor = Tensor
    tensor = staticmethod(tensor)
    
    @staticmethod
    def device(device_type):
        """Return a dummy device object."""
        return device_type
    
    @staticmethod
    def from_numpy(array):
        """Convert numpy array to dummy tensor."""
        return Tensor(array)
    
    @staticmethod
    def no_grad():
        """Return a no_grad context manager."""
        return NoGradContext()
    
    @staticmethod
    def argmax(tensor_obj, dim=None, keepdim=False):
        """Return dummy argmax result."""
        import numpy as np
        if hasattr(tensor_obj, 'data') and tensor_obj.data is not None:
            result = np.argmax(tensor_obj.data, axis=dim)
            if keepdim and dim is not None:
                result = np.expand_dims(result, axis=dim)
            return Tensor(result.astype(np.int64))
        return Tensor(np.array([0]))
    
    @staticmethod
    def max(tensor_obj, dim=None, keepdim=False):
        """Return dummy max result."""
        import numpy as np
        if hasattr(tensor_obj, 'data') and tensor_obj.data is not None:
            if dim is not None:
                result = np.max(tensor_obj.data, axis=dim, keepdims=keepdim)
            else:
                result = np.max(tensor_obj.data)
            return Tensor(result)
        return Tensor(np.array([0]))
    
    @staticmethod
    def min(tensor_obj, dim=None, keepdim=False):
        """Return dummy min result."""
        import numpy as np
        if hasattr(tensor_obj, 'data') and tensor_obj.data is not None:
            if dim is not None:
                result = np.min(tensor_obj.data, axis=dim, keepdims=keepdim)
            else:
                result = np.min(tensor_obj.data)
            return Tensor(result)
        return Tensor(np.array([0]))
    
    @staticmethod
    def stack(tensors, dim=0):
        """Stack tensors along a new dimension."""
        import numpy as np
        arrays = [t.data if hasattr(t, 'data') else t for t in tensors]
        result = np.stack(arrays, axis=dim)
        return Tensor(result)
    
    @staticmethod
    def cat(tensors, dim=0):
        """Concatenate tensors."""
        import numpy as np
        arrays = [t.data if hasattr(t, 'data') else t for t in tensors]
        result = np.concatenate(arrays, axis=dim)
        return Tensor(result)
    
    @staticmethod
    def ones(*size, **kwargs):
        """Create a tensor of ones."""
        import numpy as np
        if len(size) == 1 and isinstance(size[0], (list, tuple)):
            shape = size[0]
        else:
            shape = size
        return Tensor(np.ones(shape))
    
    @staticmethod
    def zeros(*size, **kwargs):
        """Create a tensor of zeros."""
        import numpy as np
        if len(size) == 1 and isinstance(size[0], (list, tuple)):
            shape = size[0]
        else:
            shape = size
        return Tensor(np.zeros(shape))
    
    @staticmethod
    def randn(*size, **kwargs):
        """Create a tensor of random values."""
        import numpy as np
        if len(size) == 1 and isinstance(size[0], (list, tuple)):
            shape = size[0]
        else:
            shape = size
        return Tensor(np.random.randn(*shape))
    
    @staticmethod
    def sigmoid(tensor_obj):
        """Apply sigmoid activation."""
        import numpy as np
        if hasattr(tensor_obj, 'data') and tensor_obj.data is not None:
            result = 1.0 / (1.0 + np.exp(-tensor_obj.data))
            return Tensor(result)
        return tensor_obj
    
    @staticmethod
    def relu(tensor_obj):
        """Apply ReLU activation."""
        import numpy as np
        if hasattr(tensor_obj, 'data') and tensor_obj.data is not None:
            result = np.maximum(0, tensor_obj.data)
            return Tensor(result)
        return tensor_obj
    
    @staticmethod
    def tanh(tensor_obj):
        """Apply tanh activation."""
        import numpy as np
        if hasattr(tensor_obj, 'data') and tensor_obj.data is not None:
            result = np.tanh(tensor_obj.data)
            return Tensor(result)
        return tensor_obj
    
    @staticmethod
    def clamp(tensor_obj, min_val=None, max_val=None):
        """Clamp tensor values."""
        import numpy as np
        if hasattr(tensor_obj, 'data') and tensor_obj.data is not None:
            result = tensor_obj.data.copy()
            if min_val is not None:
                result = np.maximum(result, min_val)
            if max_val is not None:
                result = np.minimum(result, max_val)
            return Tensor(result)
        return tensor_obj
    
    class cuda:
        @staticmethod
        def is_available():
            return False
        
        @staticmethod
        def get_device_name(i):
            return "CPU"

logger.debug("Torch fallback module loaded - using dummy implementations")
