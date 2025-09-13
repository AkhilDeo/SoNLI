#!/usr/bin/env python3
"""
Test script for running the dynamic experiment with a small number of eval samples.
This is perfect for testing the pipeline before running the full experiment.
"""

import subprocess
import sys
import os

def main():
    """Run the experiment with eval split and limited samples for testing."""
    
    print("=" * 60)
    print("Running Dynamic Experiment Test")
    print("=" * 60)
    print()
    
    # Change to the SoNLI directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Test configuration
    eval_samples = 2  # Number of samples from eval split
    models_to_test = ["gpt-4o-mini", "llama-3.3-8b-instruct"]  # Start with just 2 models for testing
    inference_method = "openrouter"  # Use OpenRouter for testing (faster setup)
    max_workers = 2  # Lower concurrency for testing
    
    print(f"Test Configuration:")
    print(f"  - Dataset: Eval split with {eval_samples} samples")
    print(f"  - Models: {', '.join(models_to_test)}")
    print(f"  - Inference method: {inference_method}")
    print(f"  - Max workers: {max_workers}")
    print()
    
    # Build command
    cmd = [
        sys.executable, 
        "src/experiments/sonli_experiment_one.py",
        "--eval-split",
        "--eval-samples", str(eval_samples),
        "--inference-method", inference_method,
        "--max-workers", str(max_workers),
        "--models"] + models_to_test
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    print("=" * 60)
    print()
    
    try:
        # Run the experiment
        result = subprocess.run(cmd, check=True, cwd=script_dir)
        
        print()
        print("=" * 60)
        print("✅ Test experiment completed successfully!")
        print()
        print("Output files should be in:")
        print(f"  - /home/adeo1/SoNLI/outputs/sonli_experiment_one_eval_{eval_samples}samples/")
        print("    ├── gpt-4o-mini/")
        print("    │   ├── results/experiment_results_TIMESTAMP.json")
        print("    │   ├── plots/judge_bayes_distributions_TIMESTAMP.png")
        print("    │   └── checkpoints/")
        print("    └── llama-3.3-8b-instruct/")
        print("        ├── results/")
        print("        ├── plots/")
        print("        └── checkpoints/")
        print()
        print("Next steps:")
        print("  1. Check the output files to verify the pipeline works")
        print("  2. Run with more models: --models gpt-4o gpt-4o-mini llama-3.3-8b-instruct deepseek-v3-chat llama-3.3-70b-instruct qwen3-32b")
        print("  3. Run with more samples: --eval-samples 10 or --eval-samples 50")
        print("  4. Run the full experiment without --eval-split")
        print("=" * 60)
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print(f"❌ Test experiment failed with exit code: {e.returncode}")
        print("Check the error messages above for debugging information.")
        print("=" * 60)
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("⚠️  Test experiment interrupted by user")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
