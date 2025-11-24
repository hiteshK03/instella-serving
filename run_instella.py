import sys
from vllm import LLM, SamplingParams
import utils
def main():
    # Setup environment
    utils.setup_environment()
    
    # Get model path (using the robust download/cache check)
    try:
        model_path = utils.get_model_path()
    except Exception as e:
        print(f"Error getting model path: {e}")
        return
    print(f"Attempting to load patched model from {model_path}...")
            
    # Use the patcher context manager
    with utils.InstellaPatcher(model_path):
        try:
            llm = LLM(model=model_path, trust_remote_code=True)
            print("Loading successful.")
            
            sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
            prompts = ["Hello, my name is", "The capital of France is"]
            outputs = llm.generate(prompts, sampling_params)
            
            for output in outputs:
                print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
                
        except Exception as e:
            print(f"Failed: {e}")
if __name__ == "__main__":
    main()
