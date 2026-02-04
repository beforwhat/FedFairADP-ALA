# models/__init__.py

from .vgg11 import VGG11
from .custom_cnn import CustomCNN
from .model_utils import (
    get_model,
    copy_model,
    average_models,
    model_to_vector,
    vector_to_model,
    get_num_parameters
)

__all__ = [
    "VGG11",
    "CustomCNN",
    "get_model",
    "copy_model",
    "average_models",
    "model_to_vector",
    "vector_to_model",
    "get_num_parameters"
]