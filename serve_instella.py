import sys
import uvloop
import utils
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser

def main():
    # Setup environment
    utils.setup_environment()
    
    # Get model path
    try:
        model_path = utils.get_model_path()
    except Exception as e:
        print(f"Error getting model path: {e}")
        return

    # Prepare args
    # We inject defaults into sys.argv if they are not provided by the user.
    # This allows the user to override them if needed.
    
    # Check if --model is present
    has_model = False
    for arg in sys.argv:
        if arg == "--model" or arg.startswith("--model="):
            has_model = True
            break
    
    if not has_model:
        sys.argv.extend(["--model", model_path])
        
    # Check if --served-model-name is present
    has_name = False
    for arg in sys.argv:
        if arg == "--served-model-name" or arg.startswith("--served-model-name="):
            has_name = True
            break
            
    if not has_name:
        sys.argv.extend(["--served-model-name", "instella"])
        
    # Check if --trust-remote-code is present
    if "--trust-remote-code" not in sys.argv:
        sys.argv.append("--trust-remote-code")

    print(f"Serving Instella model from: {model_path}")
    print(f"Effective args: {sys.argv[1:]}")

    # Apply patches using the context manager.
    # This patches the config loading in-memory (transformers/vllm) 
    # and modifies the olmo2.py file on disk for workers.
    with utils.InstellaPatcher(model_path):
        parser = FlexibleArgumentParser(
            description="vLLM OpenAI-Compatible RESTful API server."
        )
        parser = make_arg_parser(parser)
        args = parser.parse_args()
        validate_parsed_serve_args(args)

        # Run the server
        uvloop.run(run_server(args))

if __name__ == "__main__":
    main()
