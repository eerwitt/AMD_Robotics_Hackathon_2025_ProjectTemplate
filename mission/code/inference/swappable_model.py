import gc
import torch
import logging
from smolagents import TransformersModel


class SwappableTransformersModel(TransformersModel):
    """
    A wrapper that allows the underlying LLM to be moved to CPU (offloaded)
    and back to GPU (reloaded) on demand while retaining any quantization config.
    """
    def __init__(self, model_id, quantization_config=None, device_map="auto", **kwargs):
        self.model_id = model_id
        self.quantization_config = quantization_config
        self.device_map = device_map

        # Capture any model kwargs that should be forwarded to Hugging Face
        model_kwargs = dict(kwargs.pop("model_kwargs", {}) or {})
        if quantization_config is not None:
            model_kwargs.setdefault("quantization_config", quantization_config)

        # Initialize parent with retained quantization/model kwargs
        super().__init__(
            model_id=model_id,
            device_map=device_map,
            model_kwargs=model_kwargs,
            **kwargs,
        )
        self.is_loaded = True

    def offload(self):
        """Moves the model to CPU to free up VRAM for tools."""
        if not self.is_loaded:
            return

        logging.info("📉 [Agent] Offloading Main LLM to CPU...")
        
        # Move model to CPU
        if hasattr(self, 'model'):
            self.model.to("cpu")
        
        # Clear Cache
        torch.cuda.empty_cache()
        gc.collect()
        self.is_loaded = False
        
        logging.info(f"✅ [Agent] Offloaded. VRAM Free: {torch.cuda.memory_allocated()/1024**3:.2f} GB used.")

    def reload(self):
        """Moves the model back to GPU for inference."""
        if self.is_loaded:
            return

        logging.info("📈 [Agent] Reloading Main LLM to GPU...")
        
        # Move model back to CUDA
        if hasattr(self, 'model'):
            self.model.to("cuda")
            
        self.is_loaded = True
        logging.info("✅ [Agent] Reloaded.")
