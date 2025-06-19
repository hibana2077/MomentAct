"""
Model creation and activation utilities.
"""
import torch.nn as nn
import timm
from acts.momentact import MomentAct


def create_model_with_activation(cfg, num_classes):
    """
    Create a model with the specified activation function.
    
    Args:
        cfg: Configuration dictionary
        num_classes: Number of output classes
        
    Returns:
        torch.nn.Module: The model with custom activation
    """
    model_name = cfg['model']['name']
    pretrained = cfg['model'].get('pretrained', False)
    activation_type = cfg['activation']['type']
    
    # Get activation layer
    act_layer = get_activation_layer(activation_type)
    
    # Create model with specified activation
    model = timm.create_model(
        model_name, 
        pretrained=pretrained, 
        num_classes=num_classes,
        act_layer=act_layer
    )
    
    return model


def get_activation_layer(activation_type):
    """
    Get the activation layer class for the specified type.
    
    Args:
        activation_type: Type of activation to use
        
    Returns:
        Activation layer class
    """
    activation_map = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'gelu': nn.GELU,
        'swish': nn.SiLU,
        'moment': MomentAct,
    }
    
    return activation_map.get(activation_type, nn.ReLU)
