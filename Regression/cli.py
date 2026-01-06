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
            # Get data config
            if hasattr(self.config, 'data') and hasattr(self.config.data, 'init_args'):
                data_init_args = self.config.data.init_args
                
                # Convert Namespace to dict
                if hasattr(data_init_args, '__dict__'):
                    data_config_dict = vars(data_init_args).copy()
                elif isinstance(data_init_args, dict):
                    data_config_dict = data_init_args.copy()
                else:
                    # Try to get all attributes
                    data_config_dict = {}
                    for key in dir(data_init_args):
                        if not key.startswith('_') and not callable(getattr(data_init_args, key, None)):
                            try:
                                value = getattr(data_init_args, key)
                                data_config_dict[key] = value
                            except:
                                pass
                
                # Check if model needs input_dim
                if (hasattr(self.config, 'model') and 
                    hasattr(self.config.model, 'init_args') and
                    hasattr(self.config.model.init_args, 'model') and
                    hasattr(self.config.model.init_args.model, 'init_args')):
                    
                    model_model_init_args = self.config.model.init_args.model.init_args
                    current_input_dim = getattr(model_model_init_args, 'input_dim', None)
                    
                    # If input_dim is 0 or None, compute from datamodule
                    if current_input_dim is None or current_input_dim == 0:
                        # Create temporary datamodule to get input_dim
                        from Regression.datamodule import DataModuleRGS
                        
                        # Create and setup datamodule
                        temp_dm = DataModuleRGS(**data_config_dict)
                        temp_dm.setup('fit')
                        computed_input_dim = temp_dm.get_input_dim()
                        
                        # Set input_dim in model config
                        setattr(model_model_init_args, 'input_dim', computed_input_dim)
                        
        except Exception as e:
            # If auto-detection fails, we'll try in after_instantiate_classes
            import warnings
            warnings.warn(
                f"Could not auto-detect input_dim in before_instantiate_classes: {e}. "
                "Will try again after instantiation."
            )
    
    def after_instantiate_classes(self):
        """
        Called after instantiating classes.
        Fallback: Auto-sets input_dim in model from datamodule if not set.
        """
        try:
            # If model's input_dim is still 0, get it from datamodule
            if (hasattr(self.model, 'model') and 
                hasattr(self.model.model, 'input_dim') and
                self.model.model.input_dim == 0):
                
                if hasattr(self.datamodule, 'get_input_dim'):
                    # Setup datamodule if not already done
                    if not hasattr(self.datamodule, 'input_dim') or self.datamodule.input_dim is None:
                        self.datamodule.setup('fit')
                    
                    input_dim = self.datamodule.get_input_dim()
                    
                    # Get model config from config object (not from partially built model)
                    if (hasattr(self.config, 'model') and 
                        hasattr(self.config.model, 'init_args') and
                        hasattr(self.config.model.init_args, 'model') and
                        hasattr(self.config.model.init_args.model, 'init_args')):
                        
                        model_init_args = self.config.model.init_args.model.init_args
                        
                        # Get config values
                        hidden_layers = getattr(model_init_args, 'hidden_layers', [128, 64, 32])
                        dropout_rates = getattr(model_init_args, 'dropout_rates', [0.15, 0.1, 0.05])
                        activation = getattr(model_init_args, 'activation', 'relu')
                        output_dim = getattr(model_init_args, 'output_dim', 1)
                        
                        # Recreate model with correct input_dim
                        from Regression.modelfactory import RegressionModel
                        
                        model_config = {
                            'input_dim': input_dim,
                            'hidden_layers': hidden_layers,
                            'dropout_rates': dropout_rates,
                            'activation': activation,
                            'output_dim': output_dim,
                        }
                        
                        # Create new model with correct input_dim
                        new_model = RegressionModel(**model_config)
                        self.model.model = new_model
                    else:
                        # Fallback: use set_input_dim if available
                        if hasattr(self.model.model, 'set_input_dim'):
                            self.model.model.set_input_dim(input_dim)
                        else:
                            raise RuntimeError("Cannot set input_dim: model config not accessible")
                    
        except Exception as e:
            import warnings
            import traceback
            warnings.warn(
                f"Could not auto-set input_dim in after_instantiate_classes: {e}\n"
                f"Traceback: {traceback.format_exc()}\n"
                "Please set input_dim manually in config."
            )

