"""
Custom Lightning CLI for image classification tasks.

This module extends LightningCLI to add custom arguments for checkpoint
management and resume training, and links num_classes from datamodule to model.
"""

import lightning as L
from lightning.pytorch.cli import LightningCLI


class IMGLightningCLI(LightningCLI):
    """
    Custom Lightning CLI for image classification tasks.
    
    Extends LightningCLI with additional arguments for:
    - Resume training from checkpoint
    - Selecting checkpoint for testing (best/last)
    - Auto-linking num_classes from datamodule to model
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
        Auto-sets num_classes in model config from datamodule.
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
                
                # Check if model needs num_classes
                if (hasattr(self.config, 'model') and 
                    hasattr(self.config.model, 'init_args') and
                    hasattr(self.config.model.init_args, 'model') and
                    hasattr(self.config.model.init_args.model, 'init_args')):
                    
                    model_model_init_args = self.config.model.init_args.model.init_args
                    current_num_classes = getattr(model_model_init_args, 'num_classes', None)
                    
                    # If num_classes is 0 or None, compute from datamodule
                    if current_num_classes is None or current_num_classes == 0:
                        # Create temporary datamodule to get dimensions
                        from ImageClassification.datamodule import DataModuleIMG
                        
                        # Create and setup datamodule
                        temp_dm = DataModuleIMG(**data_config_dict)
                        temp_dm.setup('fit')
                        
                        computed_num_classes = temp_dm.get_num_classes()
                        
                        # Set num_classes in model config
                        setattr(model_model_init_args, 'num_classes', computed_num_classes)
                        
        except Exception as e:
            # If auto-detection fails, we'll try in after_instantiate_classes
            import warnings
            warnings.warn(
                f"Could not auto-detect num_classes in before_instantiate_classes: {e}. "
                "Will try again after instantiation."
            )
    
    def after_instantiate_classes(self):
        """
        Called after instantiating classes.
        Fallback: Auto-sets num_classes in model from datamodule if not set.
        """
        try:
            # If model's num_classes is still 0, get it from datamodule
            if (hasattr(self.model, 'model') and 
                hasattr(self.model.model, 'num_classes')):
                
                num_classes_ok = self.model.model.num_classes > 0
                
                if not num_classes_ok:
                    if hasattr(self.datamodule, 'get_num_classes'):
                        # Setup datamodule if not already done
                        if not hasattr(self.datamodule, 'num_classes') or self.datamodule.num_classes is None:
                            self.datamodule.setup('fit')
                        
                        num_classes = self.datamodule.get_num_classes()
                        
                        # Get model config from config object
                        if (hasattr(self.config, 'model') and 
                            hasattr(self.config.model, 'init_args') and
                            hasattr(self.config.model.init_args, 'model') and
                            hasattr(self.config.model.init_args.model, 'init_args')):
                            
                            model_init_args = self.config.model.init_args.model.init_args
                            
                            # Get config values
                            input_channels = getattr(model_init_args, 'input_channels', 3)
                            architecture = getattr(model_init_args, 'architecture', 'medium')
                            conv_layers = getattr(model_init_args, 'conv_layers', None)
                            hidden_dims = getattr(model_init_args, 'hidden_dims', None)
                            dropout_rates = getattr(model_init_args, 'dropout_rates', None)
                            activation = getattr(model_init_args, 'activation', 'relu')
                            input_size = getattr(self.datamodule, 'image_size', (224, 224))
                            
                            # Recreate model with correct num_classes
                            from ImageClassification.modelfactory import ImageClassificationModel
                            
                            model_config = {
                                'input_channels': input_channels,
                                'num_classes': num_classes,
                                'architecture': architecture,
                                'conv_layers': conv_layers,
                                'hidden_dims': hidden_dims,
                                'dropout_rates': dropout_rates,
                                'activation': activation,
                                'input_size': input_size,
                            }
                            
                            # Create new model with correct num_classes
                            new_model = ImageClassificationModel(**model_config)
                            self.model.model = new_model
                        else:
                            # Fallback: use set_num_classes if available
                            if hasattr(self.model.model, 'set_num_classes'):
                                self.model.model.set_num_classes(num_classes)
                            else:
                                raise RuntimeError("Cannot set num_classes: model config not accessible")
                    
        except Exception as e:
            import warnings
            import traceback
            warnings.warn(
                f"Could not auto-set num_classes in after_instantiate_classes: {e}\n"
                f"Traceback: {traceback.format_exc()}\n"
                "Please set num_classes manually in config."
            )

