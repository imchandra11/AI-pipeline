"""
Custom Lightning CLI for regression tasks.

This module extends LightningCLI to add custom arguments for checkpoint
management and resume training, and links input_dim from datamodule to model.
"""

import lightning as L
from lightning.pytorch.cli import LightningCLI


class RGSLightningCLI(LightningCLI):
    """
    Custom Lightning CLI for regression tasks.
    
    Extends LightningCLI with additional arguments for:
    - Resume training from checkpoint
    - Selecting checkpoint for testing (best/last)
    - Auto-linking input_dim from datamodule to model
    """
    
    def add_arguments_to_parser(self, parser):
        """
        Add custom arguments to the parser.
        
        Args:
            parser: Argument parser
        """
        # For RESUME training
        parser.add_argument("--fit.ckpt_path", type=str, default=None)
        
        # Select last or best checkpoint for testing
        parser.add_argument("--test.ckpt_path", type=str, default="best")
    
    def before_instantiate_classes(self):
        """
        Called before instantiating classes.
        Auto-sets input_dim in model config from datamodule.
        """
        try:
            # Access config - Lightning CLI uses Namespace-like objects
            # We need to traverse: config.model.init_args.model.init_args.input_dim
            config_dict = self.config
            
            # Helper to safely get nested value
            def get_nested_value(obj, path):
                """Get nested value from config object."""
                current = obj
                for key in path:
                    if isinstance(current, dict):
                        current = current.get(key)
                    else:
                        current = getattr(current, key, None)
                    if current is None:
                        return None
                return current
            
            # Helper to set nested value
            def set_nested_value(obj, path, value):
                """Set nested value in config object."""
                current = obj
                for key in path[:-1]:
                    if isinstance(current, dict):
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    else:
                        if not hasattr(current, key):
                            setattr(current, key, type(current)())
                        current = getattr(current, key)
                
                # Set final value
                if isinstance(current, dict):
                    current[path[-1]] = value
                else:
                    setattr(current, path[-1], value)
            
            # Check if input_dim needs to be set
            input_dim_path = ['model', 'init_args', 'model', 'init_args', 'input_dim']
            current_input_dim = get_nested_value(config_dict, input_dim_path)
            
            # If input_dim is 0 or None, compute from datamodule
            if current_input_dim is None or current_input_dim == 0:
                # Get data config
                data_path = ['data', 'init_args']
                data_init_args = get_nested_value(config_dict, data_path)
                
                if data_init_args:
                    # Create temporary datamodule to get input_dim
                    from Regression.datamodule import DataModuleRGS
                    import copy
                    
                    # Convert to dict if needed
                    if not isinstance(data_init_args, dict):
                        data_config_dict = {k: getattr(data_init_args, k) for k in dir(data_init_args) 
                                          if not k.startswith('_')}
                    else:
                        data_config_dict = copy.deepcopy(data_init_args)
                    
                    # Create and setup datamodule
                    temp_dm = DataModuleRGS(**data_config_dict)
                    temp_dm.setup('fit')
                    computed_input_dim = temp_dm.get_input_dim()
                    
                    # Set input_dim in model config
                    set_nested_value(config_dict, input_dim_path, computed_input_dim)
                    
        except Exception as e:
            # If auto-detection fails, user must set input_dim manually
            import warnings
            warnings.warn(
                f"Could not auto-detect input_dim: {e}. "
                "Please set input_dim manually in config."
            )

