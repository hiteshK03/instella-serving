import os
import torch
import re
from contextlib import contextmanager
import transformers
from transformers import AutoConfig

def setup_environment():
    # Set environment variables
    # Instella has head size 80 (2560/32). ROCm backend default validation rejects 80.
    # We try to force Flash Attention or Flex Attention.
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLEX_ATTENTION"
    pass

def get_model_path():
    # Return the model ID
    return "amd/Instella-3B"

@contextmanager
def InstellaPatcher(model_path):
    """
    Patches vLLM's Olmo2 implementation to support Instella's architecture (Pre-Norm).
    Also patches the config loading to make vLLM treat Instella as Olmo2.
    
    Since vLLM spawns worker processes, in-memory monkey patching of the model architecture
    won't propagate to workers. We must modify the library file on disk temporarily.
    """
    import vllm.model_executor.models.olmo2 as olmo2_module
    import vllm.transformers_utils.config as config_module
    from transformers import Olmo2Config
    
    # File patching logic
    olmo2_file_path = olmo2_module.__file__
    with open(olmo2_file_path, 'r') as f:
        original_content = f.read()
        
    # Define PyTorchRMSNorm and inject it
    pytorch_rmsnorm_code = """
class PyTorchRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        weight_dtype = dtype or torch.get_default_dtype()
        self.has_weight = has_weight
        self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.has_weight:
            return self.weight * hidden_states.to(input_dtype)
        return hidden_states.to(input_dtype)

# Replace RMSNorm with PyTorchRMSNorm for Instella compatibility check
RMSNorm = PyTorchRMSNorm
"""

    # Insert PyTorchRMSNorm before Olmo2Attention class definition
    pattern_attn = r"class Olmo2Attention\(nn\.Module\):"
    match_attn = re.search(pattern_attn, original_content)
    
    if match_attn:
        # Insert our RMSNorm override before Olmo2Attention
        # Note: We insert it after imports (which are at top) and before the first class that uses it
        intermediate_content = original_content[:match_attn.start()] + pytorch_rmsnorm_code + "\n\n" + original_content[match_attn.start():]
    else:
        print("WARNING: Could not find Olmo2Attention class start. RMSNorm override might fail.")
        intermediate_content = original_content

    # Define the new Olmo2DecoderLayer class code
    # We construct it as a string to inject into the file.
    # Note: We use names available in olmo2.py (e.g. Olmo2Attention, Olmo2MLP, RMSNorm)
    # RMSNorm here will refer to our PyTorchRMSNorm because we injected the override above.
    new_decoder_layer_code = """
class Olmo2DecoderLayer(nn.Module):
    \"\"\"
    Patched for Instella (Pre-Norm) via runtime file modification.
    \"\"\"
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        assert isinstance(config, (Olmo2Config, Olmo3Config))
        # Attention block.
        self.self_attn = Olmo2Attention(
            vllm_config=vllm_config, prefix=f"{prefix}.self_attn"
        )

        # MLP block.
        self.mlp = Olmo2MLP(vllm_config=vllm_config, prefix=f"{prefix}.mlp")

        # LayerNorm - Instella uses pre_* names in weights
        self.pre_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-Norm Architecture: x + Attn(LN(x))
        
        # Attention block.
        residual = hidden_states
        hidden_states = self.pre_attention_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = hidden_states + residual

        # MLP block.
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states
"""

    # Regex to replace the existing class Olmo2DecoderLayer(nn.Module): ... up to the next class
    # The original class ends before "class Olmo2Model"
    pattern_decoder = r"class Olmo2DecoderLayer\(nn\.Module\):.*?(?=\nclass Olmo2Model)"
    
    # Check if we can find the pattern
    match_decoder = re.search(pattern_decoder, intermediate_content, re.DOTALL)
    if not match_decoder:
        print("WARNING: Could not find Olmo2DecoderLayer class in olmo2.py to patch. Model loading might fail.")
        patched_content = intermediate_content
    else:
        print(f"Patching {olmo2_file_path} on disk (RMSNorm override + Pre-Norm)...")
        patched_content = intermediate_content[:match_decoder.start()] + new_decoder_layer_code + "\n\n" + intermediate_content[match_decoder.end():]

    # Write the patched content
    try:
        with open(olmo2_file_path, 'w') as f:
            f.write(patched_content)
    except PermissionError:
        print(f"ERROR: Cannot write to {olmo2_file_path}. Please run with appropriate permissions.")
        raise

    # Config patching (in-memory is fine as it propagates via pickling of the config object)
    original_get_config = config_module.get_config
    original_from_pretrained = transformers.AutoConfig.from_pretrained
    
    # Helper to modify config
    def _modify_config(config):
        # Check if this is our Instella model
        is_instella = False
        if hasattr(config, "architectures") and config.architectures and "InstellaForCausalLM" in config.architectures:
            is_instella = True
        elif hasattr(config, "model_type") and config.model_type == "instella":
            is_instella = True
            
        if is_instella:
            print("Detected InstellaForCausalLM/Config. Masking as Olmo2ForCausalLM for vLLM compatibility.")
            
            # 1. Mask Architecture
            config.architectures = ["Olmo2ForCausalLM"]
            
            # 2. Add rope_parameters if missing (needed by Olmo2 code)
            if not hasattr(config, "rope_parameters"):
                # Instella uses rope_theta = 10000.0, defaults
                config.rope_parameters = {
                    "rope_type": "default",
                    "rope_theta": getattr(config, "rope_theta", 10000.0)
                }
                
            # 3. Satisfy isinstance(config, Olmo2Config)
            if not isinstance(config, Olmo2Config):
                class PatchedInstellaConfig(config.__class__, Olmo2Config):
                    pass
                config.__class__ = PatchedInstellaConfig
                
        return config

    # Monkey patch get_config
    def patched_get_config(*args, **kwargs):
        config = original_get_config(*args, **kwargs)
        return _modify_config(config)

    config_module.get_config = patched_get_config

    # Monkey patch AutoConfig.from_pretrained
    def patched_from_pretrained(*args, **kwargs):
        config = original_from_pretrained(*args, **kwargs)
        return _modify_config(config)

    transformers.AutoConfig.from_pretrained = patched_from_pretrained

    try:
        yield
    finally:
        # Restore file content
        print(f"Restoring {olmo2_file_path}...")
        with open(olmo2_file_path, 'w') as f:
            f.write(original_content)
            
        # Restore original functions
        config_module.get_config = original_get_config
        transformers.AutoConfig.from_pretrained = original_from_pretrained
