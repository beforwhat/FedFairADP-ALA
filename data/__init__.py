# data/__init__.py

from .dataset_factory import load_dataset, get_input_shape, get_num_classes

__all__ = ["load_dataset", "get_input_shape", "get_num_classes"]