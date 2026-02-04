# models/model_utils.py

import torch
import torch.nn as nn
from typing import Dict, List, Union, Type
from .vgg11 import VGG11
from .custom_cnn import CustomCNN


def get_model(
    model_name: str,
    in_channels: int,
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create model instance.
    
    Args:
        model_name: "vgg11" or "custom_cnn"
        in_channels: Input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of output classes
    """
    if model_name.lower() == "vgg11":
        return VGG11(in_channels=in_channels, num_classes=num_classes)
    elif model_name.lower() == "custom_cnn":
        return CustomCNN(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def copy_model(model: nn.Module) -> nn.Module:
    """Create a deep copy of the model."""
    copied = type(model)(
        in_channels=model.features[0].in_channels,
        num_classes=model.classifier[-1].out_features
    )
    copied.load_state_dict(model.state_dict())
    return copied


def average_models(
    models: List[nn.Module],
    weights: List[float] = None
) -> nn.Module:
    """
    Compute weighted average of model parameters.
    
    Args:
        models: List of models to average
        weights: Optional list of weights (must sum to 1)
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    avg_model = copy_model(models[0])
    avg_dict = avg_model.state_dict()
    
    with torch.no_grad():
        for key in avg_dict.keys():
            avg_dict[key].zero_()
            for model, weight in zip(models, weights):
                avg_dict[key] += weight * model.state_dict()[key]
    
    avg_model.load_state_dict(avg_dict)
    return avg_model


def model_to_vector(model: nn.Module) -> torch.Tensor:
    """Flatten model parameters into a 1D vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def vector_to_model(vector: torch.Tensor, model: nn.Module) -> nn.Module:
    """Load parameters from a vector into a model."""
    model_copy = copy_model(model)
    pointer = 0
    with torch.no_grad():
        for param in model_copy.parameters():
            num_param = param.numel()
            param.copy_(vector[pointer:pointer + num_param].view_as(param))
            pointer += num_param
    return model_copy


def get_num_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)