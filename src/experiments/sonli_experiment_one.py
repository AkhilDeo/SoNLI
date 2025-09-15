import sys
import os
from datetime import datetime
from tqdm import tqdm
import json
import concurrent.futures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
import argparse
from typing import Dict, List, Tuple, Optional, Any
import copy

from dotenv import load_dotenv

# Load environment variables FIRST before importing modules that need them
load_dotenv()

from langchain_community.cache import SQLiteCache
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.globals import set_llm_cache

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.supporting_explanations_no_q import supporting_explanations_prompt_with_no_question
from prompts.opposing_explanations_no_q import opposing_explanations_prompt_with_no_question
from prompts.judge import scoring_prompt
from utils.openrouter_client import call_openrouter_chat_completion, deepseek_r1_completion, OpenRouterLLM

# Optional vLLM imports - will be checked during runtime
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("[WARN] vLLM not available. Install with: pip install vllm")

# --- Updated Model Configurations ---
EXPLANATION_MODELS_CONFIG = {
    "gpt-4o": {"type": "openai", "model_id": "gpt-4o"},
    "gpt-4o-mini": {"type": "openai", "model_id": "gpt-4o-mini"},
    "llama-3.1-8b-instruct": {"type": "openrouter", "model_id": "meta-llama/llama-3.1-8b-instruct"},
    "deepseek-v3-chat": {"type": "openrouter", "model_id": "deepseek/deepseek-chat-v3-0324"},
    "llama-3.1-70b-instruct": {"type": "openrouter", "model_id": "meta-llama/llama-3.1-70b-instruct"},
    "qwen3-32b": {"type": "openrouter", "model_id": "qwen/qwen3-32b"}
}

# HuggingFace model mappings for vLLM
HUGGINGFACE_MODEL_MAPPING = {
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-70b-instruct": "meta-llama/Llama-3.1-70B-Instruct", 
    "qwen3-32b": "Qwen/Qwen2.5-32B-Instruct",
    "deepseek-v3-chat": "deepseek-ai/DeepSeek-V3-Base"  # Adjust based on available model
}

# Consistent generation parameters
TEMPERATURE = 0.7
MAX_TOKENS = 5000

def clean_xml_tags(text: str) -> str:
    """Removes <think>, </think>, <answer>, </answer> tags from text, case-insensitive."""
    if not isinstance(text, str):
        return ""
    cleaned_text = re.sub(r"</?(think|answer)>", "", text, flags=re.IGNORECASE)
    return cleaned_text.strip()

def parse_judge_score(judge_response_content: str) -> int:
    """Parses the SCORE: X from the judge's response."""
    if not judge_response_content:
        return -1
    
    match = re.search(r"SCORE:\s*(\d+)", judge_response_content, re.MULTILINE | re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            print(f"[WARN] Judge score found but could not convert to int: '{match.group(1)}'")
            return -1
    else:
        numbers = re.findall(r"\b(\d+)\b", judge_response_content)
        if numbers:
            for num_str in reversed(numbers):
                try:
                    num = int(num_str)
                    if 0 <= num <= 10:
                        print(f"[INFO] Parsed judge score '{num}' using fallback regex.")
                        return num
                except ValueError:
                    continue
            print(f"[WARN] Could not parse a valid judge score (0-10) from: '{judge_response_content[:100]}...'")
        else:
            print(f"[WARN] No SCORE: X pattern or numbers found in judge response: '{judge_response_content[:100]}...'")
        return -1

# UNLIScorer removed - focusing on LLM-as-judge only for this experiment

class VLLMClient:
    """vLLM-based client for local inference."""
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not available. Please install it first.")
        
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        
        print(f"Initializing vLLM client for {model_path} with {tensor_parallel_size} GPUs...")
        # Prefer bfloat16 on A100s (ba100 partition) to reduce memory, fall back to auto if unsupported
        try:
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True,
                dtype="bfloat16",
                gpu_memory_utilization=0.90
            )
        except TypeError:
            # Older vLLM versions may not support dtype / gpu_memory_utilization args
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True
            )
        
        self.sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stop=["</s>", "<|im_end|>", "<|endoftext|>"]
        )
        
    def invoke(self, messages: List) -> Any:
        """Generate response using vLLM."""
        if not messages or not isinstance(messages, list):
            raise ValueError("Input 'messages' must be a non-empty list.")
        
        user_prompt_message = messages[-1]
        if hasattr(user_prompt_message, 'content'):
            prompt = user_prompt_message.content
        elif isinstance(user_prompt_message, str):
            prompt = user_prompt_message
        else:
            raise ValueError("Last message in 'messages' must have a 'content' attribute or be a string.")
        
        # Generate response
        outputs = self.llm.generate([prompt], self.sampling_params)
        response_text = ""
        try:
            if outputs and len(outputs) > 0 and outputs[0].outputs and len(outputs[0].outputs) > 0:
                response_text = outputs[0].outputs[0].text
        except Exception:
            response_text = ""
        
        # Return in HumanMessage format to maintain compatibility
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        return MockResponse(response_text)

def create_llm_client(model_name: str, config: Dict, inference_method: str, tensor_parallel_size: int = 1):
    """Create appropriate LLM client based on model type and inference method.
    
    OpenAI models (gpt-4o, gpt-4o-mini) ALWAYS use OpenAI API directly.
    Open source models use inference_method to decide: openrouter vs local GPU serving.
    """
    
    # OpenAI models ALWAYS use OpenAI API directly, regardless of inference_method
    if config["type"] == "openai":
        return ChatOpenAI(
            model=config["model_id"],
            temperature=TEMPERATURE,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=MAX_TOKENS
        )
    
    # For open source models, use inference_method to decide serving approach
    elif config["type"] == "openrouter":
        if inference_method == "openrouter":
            return OpenRouterLLM(model_id=config["model_id"])
        elif inference_method == "huggingface":
            if model_name not in HUGGINGFACE_MODEL_MAPPING:
                raise ValueError(f"Model '{model_name}' not available for HuggingFace inference")
            hf_model_path = HUGGINGFACE_MODEL_MAPPING[model_name]
            return VLLMClient(hf_model_path, tensor_parallel_size)
        else:
            raise ValueError(f"Unknown inference method '{inference_method}' for open source model '{model_name}'")
    
    else:
        raise ValueError(f"Unknown model type '{config['type']}' for model '{model_name}'")

def generate_single_explanation(llm_client, scene_dialogue: str, inference: str, explanation_type: str) -> str:
    """Generate a single explanation (supporting or opposing) using a given llm_client."""
    
    max_retries = 3
    default_explanation = "LLM failed to generate an explanation."

    def try_generate_explanation(prompt, retry_count=0):
        try:
            response = llm_client.invoke([HumanMessage(content=prompt)])
            explanation = response.content
            explanation = clean_xml_tags(explanation)
            
            if not explanation or explanation.isspace():
                if retry_count < max_retries:
                    print(f"Empty response received, retrying... (attempt {retry_count + 1}/{max_retries})")
                    return try_generate_explanation(prompt, retry_count + 1)
                print(f"Max retries reached, using default explanation: '{default_explanation}'")
                return default_explanation
            return explanation
        except Exception as e:
            print(f"Error generating {explanation_type} explanation: {str(e)}")
            if retry_count < max_retries:
                print(f"Retrying due to error... (attempt {retry_count + 1}/{max_retries})")
                return try_generate_explanation(prompt, retry_count + 1)
            print(f"Max retries reached after error, using default explanation: '{default_explanation}'")
            return default_explanation
    
    # Generate explanation based on type
    if explanation_type == "supporting":
        prompt_str = supporting_explanations_prompt_with_no_question(
            scene_dialogue=scene_dialogue,
            inference=inference
        )
    elif explanation_type == "opposing":
        prompt_str = opposing_explanations_prompt_with_no_question(
            scene_dialogue=scene_dialogue,
            inference=inference
        )
    else:
        raise ValueError(f"Unknown explanation_type: {explanation_type}")
    
    return try_generate_explanation(prompt_str)

def generate_explanations(llm_client, scene_dialogue: str, inference: str) -> Tuple[str, str]:
    """Generate supporting and opposing explanations using a given llm_client.
    
    DEPRECATED: This function is kept for backward compatibility.
    Use generate_single_explanation for more efficient batch processing.
    """
    supporting_explanation = generate_single_explanation(llm_client, scene_dialogue, inference, "supporting")
    opposing_explanation = generate_single_explanation(llm_client, scene_dialogue, inference, "opposing")
    return supporting_explanation, opposing_explanation

def calculate_final_bayes_score(s_plus: float, s_minus: float) -> Optional[float]:
    """Calculate the final Bayes score based on supporting and opposing scores."""
    if not (isinstance(s_plus, (int, float)) and isinstance(s_minus, (int, float))):
        return None

    s_plus = max(0.0, min(1.0, s_plus))
    s_minus = max(0.0, min(1.0, s_minus))

    numerator = s_plus * (1 - s_minus)
    denominator_part2 = (1 - s_plus) * s_minus
    
    denominator = numerator + denominator_part2
    
    if denominator == 0:
        return 0.5
    else:
        return numerator / denominator

def add_bayes_scores(data: List[Dict]) -> List[Dict]:
    """Add Bayes scores to the data structure."""
    updated_data = []
    total_explanations_processed = 0
    total_explanations_skipped = 0

    for item_index, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Warning: Item at index {item_index} is not a dictionary. Skipping.")
            updated_data.append(item)
            continue

        new_item = item.copy()
        
        if 'explanations_from_models' in item and isinstance(item['explanations_from_models'], list):
            updated_explanations = []
            for expl_index, model_explanation in enumerate(item['explanations_from_models']):
                if not isinstance(model_explanation, dict):
                    print(f"Warning: Explanation at item_index {item_index}, expl_index {expl_index} is not a dictionary.")
                    updated_explanations.append(model_explanation)
                    total_explanations_skipped += 1
                    continue

                s_plus = model_explanation.get('supporting_score_judge')
                s_minus = model_explanation.get('opposing_score_judge')
                
                new_model_explanation = model_explanation.copy()

                if s_plus is None or s_minus is None:
                    print(f"Warning: Missing scores for item {item_index}, explanation {expl_index}. Skipping Bayes calculation.")
                    total_explanations_skipped += 1
                elif not (isinstance(s_plus, (int, float)) and isinstance(s_minus, (int, float))):
                    print(f"Warning: Non-numeric scores for item {item_index}, explanation {expl_index}. Skipping Bayes calculation.")
                    total_explanations_skipped += 1
                else:
                    final_score = calculate_final_bayes_score(s_plus, s_minus)
                    new_model_explanation['final_bayes_score'] = final_score
                    total_explanations_processed += 1
                
                updated_explanations.append(new_model_explanation)
            
            new_item['explanations_from_models'] = updated_explanations
        else:
            print(f"Warning: Item at index {item_index} is missing 'explanations_from_models' list.")

        updated_data.append(new_item)

    print(f"Total explanations processed for Bayes score: {total_explanations_processed}")
    if total_explanations_skipped > 0:
        print(f"Total explanations skipped due to missing/invalid scores: {total_explanations_skipped}")

    return updated_data

def plot_score_distributions(outputs, model_name: str, model_output_dir: str, run_timestamp: str):
    """Plot distributions of supporting and opposing scores for Judge model only."""
    supporting_judge_scores = []
    opposing_judge_scores = []
    bayes_scores = []
    
    for item in outputs:
        for model_explanation_details in item.get("explanations_from_models", []):
            if model_explanation_details.get("explanation_model_name") == model_name:
                if model_explanation_details["supporting_score_judge"] != -1:
                    supporting_judge_scores.append(model_explanation_details["supporting_score_judge"])
                if model_explanation_details["opposing_score_judge"] != -1:
                    opposing_judge_scores.append(model_explanation_details["opposing_score_judge"])
                if model_explanation_details.get("final_bayes_score") is not None:
                    bayes_scores.append(model_explanation_details["final_bayes_score"])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Judge score distributions (supporting vs opposing)
    if supporting_judge_scores or opposing_judge_scores:
        if len(supporting_judge_scores) >= 2:
            sns.kdeplot(x=supporting_judge_scores, label='Supporting Judge', color='green', ax=axes[0], fill=True)
        else:
            sns.histplot(x=supporting_judge_scores, label='Supporting Judge', color='green', ax=axes[0], stat='density', bins=10, element='step')

        if len(opposing_judge_scores) >= 2:
            sns.kdeplot(x=opposing_judge_scores, label='Opposing Judge', color='red', ax=axes[0], fill=True)
        else:
            sns.histplot(x=opposing_judge_scores, label='Opposing Judge', color='red', ax=axes[0], stat='density', bins=10, element='step')

        axes[0].set_title(f'Judge Score Distributions - {model_name}')
        axes[0].set_xlabel('Judge Score (0.0-1.0)')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].set_xlim(0, 1)
        axes[0].set_xticks(np.linspace(0, 1, 11))
    else:
        axes[0].set_title(f'No Judge scores to plot - {model_name}')

    # Bayes score distribution
    if bayes_scores:
        if len(bayes_scores) >= 2:
            sns.kdeplot(x=bayes_scores, label='Final Bayes Score', color='purple', ax=axes[1], fill=True)
        else:
            sns.histplot(x=bayes_scores, label='Final Bayes Score', color='purple', ax=axes[1], stat='density', bins=10, element='step')
        axes[1].set_title(f'Final Bayes Score Distribution - {model_name}')
        axes[1].set_xlabel('Bayes Score (0.0-1.0)')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].set_xlim(0, 1)
        axes[1].set_xticks(np.linspace(0, 1, 11))
    else:
        axes[1].set_title(f'No Bayes scores to plot - {model_name}')
    
    plt.tight_layout()
    
    plot_dir = os.path.join(model_output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_file = os.path.join(plot_dir, f"judge_bayes_distributions_{run_timestamp}.png")
    plt.savefig(plot_file)
    plt.close(fig)
    print(f"Saved score distribution plot to {plot_file}")
    
    # Print summary statistics
    print(f"\nSummary Statistics for {model_name}:")
    if supporting_judge_scores:
        print(f"Judge Supporting Scores: mean={np.mean(supporting_judge_scores):.3f}, std={np.std(supporting_judge_scores):.3f}, count={len(supporting_judge_scores)}")
    if opposing_judge_scores:
        print(f"Judge Opposing Scores: mean={np.mean(opposing_judge_scores):.3f}, std={np.std(opposing_judge_scores):.3f}, count={len(opposing_judge_scores)}")
    if bayes_scores:
        print(f"Final Bayes Scores: mean={np.mean(bayes_scores):.3f}, std={np.std(bayes_scores):.3f}, count={len(bayes_scores)}")

def save_checkpoint(data: List, stage_name: str, run_timestamp: str, processed_count: int, model_output_dir: str):
    """Save checkpoint data to model-specific directory."""
    checkpoint_dir = os.path.join(model_output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file_name = f"checkpoint_{run_timestamp}_stage_{stage_name}_count_{processed_count}.json"
    checkpoint_file_path = os.path.join(checkpoint_dir, checkpoint_file_name)
    try:
        with open(checkpoint_file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Saved checkpoint to {checkpoint_file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save checkpoint {checkpoint_file_path}: {e}")

def call_model_for_judge(llm_client, prompt: str) -> Dict[str, Any]:
    """Call the model client for judge scoring, handling different client types."""
    try:
        if isinstance(llm_client, ChatOpenAI):
            # OpenAI/OpenRouter client
            messages = [HumanMessage(content=prompt)]
            response = llm_client.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            return {
                "choices": [{"message": {"content": content}}]
            }
        elif isinstance(llm_client, VLLMClient):
            # vLLM client
            response = llm_client.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, 'content') else str(response)
            return {
                "choices": [{"message": {"content": content}}]
            }
        elif isinstance(llm_client, OpenRouterLLM):
            # OpenRouter client
            messages = [HumanMessage(content=prompt)]
            response = llm_client.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            return {
                "choices": [{"message": {"content": content}}]
            }
        else:
            # Fallback: try to call as a function
            response = llm_client(prompt)
            if isinstance(response, dict) and "choices" in response:
                return response
            else:
                # Assume it's a string response
                return {
                    "choices": [{"message": {"content": str(response)}}]
                }
    except Exception as e:
        print(f"[ERROR] call_model_for_judge failed: {e}")
        return {
            "choices": [{"message": {"content": ""}}]
        }

def process_single_model(model_name: str, llm_client, inferences_data: List[Dict], 
                        max_workers: int, run_timestamp: str,
                        output_base_dir: str, checkpoint_interval: int) -> List[Dict]:
    """Process explanations and scoring for a single model."""
    
    print(f"\n=== Processing Model: {model_name} ===")
    
    # Create model-specific output directory
    model_output_dir = os.path.join(output_base_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"Model output directory: {model_output_dir}")
    
    # Stage 1a: Generate Supporting Explanations (All Concurrent)
    print(f"--- Stage 1a: Generating Supporting Explanations for {model_name} ---")
    supporting_tasks = []
    for item_idx, inference_item in enumerate(inferences_data):
        scene_dialogue = inference_item["dialogue"]
        inference = inference_item["inference"]
        supporting_tasks.append({
            "item_idx": item_idx,
            "original_item_data": inference_item,
            "model_name": model_name,
            "llm_client": llm_client,
            "scene_dialogue": scene_dialogue,
            "inference": inference
        })

    # Dictionary to store completed explanations by item_idx
    processed_entries_by_idx = {}
    stage1a_tasks_completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task_idx = {
            executor.submit(
                generate_single_explanation, 
                task["llm_client"], 
                task["scene_dialogue"], 
                task["inference"],
                "supporting"
            ): idx
            for idx, task in enumerate(supporting_tasks)
        }
        
        with tqdm(total=len(future_to_task_idx), desc=f"Generating Supporting Explanations ({model_name})") as pbar:
            for future in concurrent.futures.as_completed(future_to_task_idx):
                task_idx = future_to_task_idx[future]
                original_task_info = supporting_tasks[task_idx]
                try:
                    supporting_explanation = future.result()
                    entry = {
                        "original_item_index": original_task_info["item_idx"],
                        "original_item_data": original_task_info["original_item_data"],
                        "inference": original_task_info["inference"],
                        "explanation_model_name": original_task_info["model_name"],
                        "supporting_explanation": supporting_explanation,
                        "opposing_explanation": None,  # Will be filled in Stage 1b
                        "supporting_score_judge": -1,
                        "opposing_score_judge": -1,
                        "supporting_judge_raw_output": "",
                        "opposing_judge_raw_output": ""
                    }
                    processed_entries_by_idx[original_task_info["item_idx"]] = entry
                    stage1a_tasks_completed_count += 1

                    if checkpoint_interval > 0 and stage1a_tasks_completed_count % checkpoint_interval == 0:
                        # Convert to list for checkpoint
                        entries_list = [processed_entries_by_idx[i] for i in sorted(processed_entries_by_idx.keys())]
                        save_checkpoint(entries_list, "stage1a_partial", run_timestamp, stage1a_tasks_completed_count, model_output_dir)
                except Exception as e:
                    print(f"[ERROR] Stage 1a: Error generating supporting explanation for item index {original_task_info['item_idx']}, model {original_task_info['model_name']}: {e}")
                pbar.update(1)

    print(f"Stage 1a Complete for {model_name}: Generated supporting explanations for {len(processed_entries_by_idx)} items.")

    # Stage 1b: Generate Opposing Explanations (All Concurrent)
    print(f"--- Stage 1b: Generating Opposing Explanations for {model_name} ---")
    opposing_tasks = []
    for item_idx, inference_item in enumerate(inferences_data):
        if item_idx in processed_entries_by_idx:  # Only process items that have supporting explanations
            scene_dialogue = inference_item["dialogue"]
            inference = inference_item["inference"]
            opposing_tasks.append({
                "item_idx": item_idx,
                "original_item_data": inference_item,
                "model_name": model_name,
                "llm_client": llm_client,
                "scene_dialogue": scene_dialogue,
                "inference": inference
            })

    stage1b_tasks_completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task_idx = {
            executor.submit(
                generate_single_explanation, 
                task["llm_client"], 
                task["scene_dialogue"], 
                task["inference"],
                "opposing"
            ): idx
            for idx, task in enumerate(opposing_tasks)
        }
        
        with tqdm(total=len(future_to_task_idx), desc=f"Generating Opposing Explanations ({model_name})") as pbar:
            for future in concurrent.futures.as_completed(future_to_task_idx):
                task_idx = future_to_task_idx[future]
                original_task_info = opposing_tasks[task_idx]
                try:
                    opposing_explanation = future.result()
                    # Update the existing entry with opposing explanation
                    item_idx = original_task_info["item_idx"]
                    if item_idx in processed_entries_by_idx:
                        processed_entries_by_idx[item_idx]["opposing_explanation"] = opposing_explanation
                        stage1b_tasks_completed_count += 1

                        if checkpoint_interval > 0 and stage1b_tasks_completed_count % checkpoint_interval == 0:
                            # Convert to list for checkpoint
                            entries_list = [processed_entries_by_idx[i] for i in sorted(processed_entries_by_idx.keys())]
                            save_checkpoint(entries_list, "stage1b_partial", run_timestamp, stage1b_tasks_completed_count, model_output_dir)
                except Exception as e:
                    print(f"[ERROR] Stage 1b: Error generating opposing explanation for item index {original_task_info['item_idx']}, model {original_task_info['model_name']}: {e}")
                pbar.update(1)

    # Convert back to list for further processing
    all_processed_entries = [processed_entries_by_idx[i] for i in sorted(processed_entries_by_idx.keys())]
    
    if all_processed_entries:
        save_checkpoint(all_processed_entries, "stage1_complete", run_timestamp, len(all_processed_entries), model_output_dir)
    print(f"Stage 1 Complete for {model_name}: Generated explanations for {len(all_processed_entries)} items.")

    # Stage 2a: Score Supporting Explanations with Judge Model (All Concurrent)
    print(f"--- Stage 2a: Scoring Supporting Explanations with Judge Model for {model_name} ---")
    if not all_processed_entries:
        print(f"No explanations to score with Judge for {model_name}. Skipping Stage 2.")
    else:
        try:
            stage2a_tasks_completed_count = 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures_supporting = []
                for entry_idx, entry in enumerate(all_processed_entries):
                    if entry["supporting_explanation"] and entry["supporting_explanation"] != "LLM failed to generate an explanation.":
                        judge_prompt_support = scoring_prompt(explanation=entry["supporting_explanation"], inference=entry["inference"])
                        future_support = executor.submit(call_model_for_judge, llm_client, judge_prompt_support)
                        futures_supporting.append({"entry_idx": entry_idx, "future": future_support})

                with tqdm(total=len(futures_supporting), desc=f"Scoring Supporting Explanations ({model_name})") as pbar_supporting:
                    future_to_entry_idx = {
                        future_info["future"]: future_info["entry_idx"]
                        for future_info in futures_supporting
                    }
                    for future in concurrent.futures.as_completed(future_to_entry_idx.keys()):
                        entry_idx = future_to_entry_idx[future]
                        entry_to_update = all_processed_entries[entry_idx]
                        try:
                            judge_response_data = future.result()
                            judge_response_content = judge_response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                            
                            content_for_parsing = clean_xml_tags(judge_response_content)
                            parsed_score = parse_judge_score(content_for_parsing)
                            
                            normalized_score = parsed_score / 10.0 if parsed_score != -1 else -1
                            
                            entry_to_update["supporting_score_judge"] = normalized_score
                            entry_to_update["supporting_judge_raw_output"] = judge_response_content
                            stage2a_tasks_completed_count += 1

                            if checkpoint_interval > 0 and stage2a_tasks_completed_count % checkpoint_interval == 0:
                                save_checkpoint(all_processed_entries, "stage2a_partial", run_timestamp, stage2a_tasks_completed_count, model_output_dir)
                        except Exception as e:
                            item_idx = entry_to_update["original_item_index"]
                            print(f"[ERROR] Stage 2a: Error judge scoring supporting explanation for item index {item_idx}, model {model_name}: {e}")
                        finally:
                            pbar_supporting.update(1)

            print(f"Stage 2a Complete for {model_name}: Judge scoring of supporting explanations finished.")

            # Stage 2b: Score Opposing Explanations with Judge Model (All Concurrent)
            print(f"--- Stage 2b: Scoring Opposing Explanations with Judge Model for {model_name} ---")
            stage2b_tasks_completed_count = 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures_opposing = []
                for entry_idx, entry in enumerate(all_processed_entries):
                    if entry["opposing_explanation"] and entry["opposing_explanation"] != "LLM failed to generate an explanation.":
                        judge_prompt_oppose = scoring_prompt(explanation=entry["opposing_explanation"], inference=entry["inference"])
                        future_oppose = executor.submit(call_model_for_judge, llm_client, judge_prompt_oppose)
                        futures_opposing.append({"entry_idx": entry_idx, "future": future_oppose})

                with tqdm(total=len(futures_opposing), desc=f"Scoring Opposing Explanations ({model_name})") as pbar_opposing:
                    future_to_entry_idx = {
                        future_info["future"]: future_info["entry_idx"]
                        for future_info in futures_opposing
                    }
                    for future in concurrent.futures.as_completed(future_to_entry_idx.keys()):
                        entry_idx = future_to_entry_idx[future]
                        entry_to_update = all_processed_entries[entry_idx]
                        try:
                            judge_response_data = future.result()
                            judge_response_content = judge_response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                            
                            content_for_parsing = clean_xml_tags(judge_response_content)
                            parsed_score = parse_judge_score(content_for_parsing)
                            
                            normalized_score = parsed_score / 10.0 if parsed_score != -1 else -1
                            
                            entry_to_update["opposing_score_judge"] = normalized_score
                            entry_to_update["opposing_judge_raw_output"] = judge_response_content
                            stage2b_tasks_completed_count += 1

                            if checkpoint_interval > 0 and stage2b_tasks_completed_count % checkpoint_interval == 0:
                                save_checkpoint(all_processed_entries, "stage2b_partial", run_timestamp, stage2b_tasks_completed_count, model_output_dir)
                        except Exception as e:
                            item_idx = entry_to_update["original_item_index"]
                            print(f"[ERROR] Stage 2b: Error judge scoring opposing explanation for item index {item_idx}, model {model_name}: {e}")
                        finally:
                            pbar_opposing.update(1)
                            
        except KeyboardInterrupt:
            print("\n[WARN] KeyboardInterrupt received during Stage 2. Attempting graceful shutdown...")
            # No direct future cancellation in ThreadPoolExecutor; proceed to save partials
            try:
                save_checkpoint(all_processed_entries, "stage2_partial_interrupt", run_timestamp, len(all_processed_entries), model_output_dir)
                print("[INFO] Saved partial checkpoint after interrupt.")
            except Exception as e:
                print(f"[WARN] Failed to save partial checkpoint after interrupt: {e}")
            raise
        
        if all_processed_entries:
            save_checkpoint(all_processed_entries, "stage2_complete", run_timestamp, len(all_processed_entries), model_output_dir)
        print(f"Stage 2 Complete for {model_name}: Judge scoring finished.")

    # Stage 3: UNLI scoring removed - focusing on LLM-as-judge only
    print(f"--- Stage 3: Skipped UNLI scoring (using LLM-as-judge only) for {model_name} ---")

    return all_processed_entries

def main():
    parser = argparse.ArgumentParser(description="Run social reasoning experiment with dynamic inference methods.")
    parser.add_argument("-l", "--limit", type=int, default=None, help="Maximum number of input items to process")
    parser.add_argument("-ci", "--checkpoint-interval", type=int, default=50, help="Checkpoint interval during generation")
    parser.add_argument("-mw", "--max-workers", type=int, default=3, help="Maximum number of worker threads")
    parser.add_argument("-im", "--inference-method", choices=["openrouter", "huggingface"], default="openrouter", 
                        help="Inference method: openrouter or huggingface")
    parser.add_argument("-gpu", "--num-gpus", type=int, default=1, help="Number of GPUs for vLLM (only for huggingface method)")
    parser.add_argument("-m", "--models", nargs="+", choices=list(EXPLANATION_MODELS_CONFIG.keys()), 
                        default=list(EXPLANATION_MODELS_CONFIG.keys()), help="Models to test")
    parser.add_argument("-t", "--test", action="store_true", help="Run with only first 3 items for testing")
    parser.add_argument("--eval-split", action="store_true", help="Use eval split dataset instead of main split")
    parser.add_argument("--eval-samples", type=int, default=0, help="Number of samples to take from eval split (0 = all)")
    
    args = parser.parse_args()

    # Initialize
    print("Initializing experiment...")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.inference_method == "huggingface" and not VLLM_AVAILABLE:
        print("[ERROR] vLLM is required for huggingface inference method but not available.")
        return
    
    # Load data
    print("Loading input data...")
    if args.eval_split:
        input_file = "datasets/socialnli/socialnli_human_eval_split.json"
        print("[INFO] Using eval split dataset")
    else:
        input_file = "datasets/socialnli/socialnli_main_split_not_scored.json"
        print("[INFO] Using main split dataset")
    
    try:
        with open(input_file, 'r') as f:
            inferences_data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Input file '{input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"[ERROR] Could not decode JSON from '{input_file}'.")
        return
        
    print(f"Loaded {len(inferences_data)} inference items from {input_file}")

    # Apply sampling, test mode, or limit
    if args.eval_split:
        # Sample from eval split
        import random
        if args.eval_samples > 0 and args.eval_samples < len(inferences_data):
            print(f"[INFO] Sampling {args.eval_samples} items from eval split")
            random.seed(42)  # For reproducible sampling
            inferences_data = random.sample(inferences_data, args.eval_samples)
        else:
            print(f"[INFO] Using all {len(inferences_data)} items from eval split")
    elif args.test:
        print("[TEST MODE] Using only first 3 items")
        inferences_data = inferences_data[:3]
    elif args.limit is not None:
        if args.limit > 0 and args.limit < len(inferences_data):
            print(f"[INFO] --limit set to {args.limit}. Processing only the first {args.limit} items.")
            inferences_data = inferences_data[:args.limit]

    # Set up timestamped output directory
    if args.eval_split:
        output_base_dir = f"outputs/exp_one_eval_{run_timestamp}_{args.eval_samples}_samples"
    else:
        output_base_dir = f"outputs/exp_one_{run_timestamp}"
    os.makedirs(output_base_dir, exist_ok=True)

    # Initialize language cache (clear previous cache so each run starts fresh)
    cache_db_path = ".langchain_sonli_exp1_cache.db"
    try:
        if os.path.exists(cache_db_path):
            os.remove(cache_db_path)
            print(f"[INFO] Cleared existing cache at {cache_db_path}")
        # Also remove SQLite sidecar files if present
        for sidecar_suffix in ("-wal", "-shm"):
            sidecar_path = f"{cache_db_path}{sidecar_suffix}"
            if os.path.exists(sidecar_path):
                os.remove(sidecar_path)
                print(f"[INFO] Cleared existing cache sidecar at {sidecar_path}")
    except Exception as e:
        print(f"[WARN] Could not clear cache files: {e}")
    set_llm_cache(SQLiteCache(database_path=cache_db_path))

    # Process each model separately
    for model_name in args.models:
        if model_name not in EXPLANATION_MODELS_CONFIG:
            print(f"[WARN] Unknown model '{model_name}'. Skipping.")
            continue

        config = EXPLANATION_MODELS_CONFIG[model_name]
        
        # Check if model is available for the chosen inference method
        if args.inference_method == "huggingface" and config["type"] == "openrouter" and model_name not in HUGGINGFACE_MODEL_MAPPING:
            print(f"[WARN] Open source model '{model_name}' not available for HuggingFace inference. Skipping.")
            continue
        
        try:
            # Determine actual method used
            if config["type"] == "openai":
                actual_method = "OpenAI API (direct)"
            elif config["type"] == "openrouter":
                actual_method = f"{args.inference_method} (open source)"
            else:
                actual_method = args.inference_method
                
            print(f"\nInitializing {model_name} ({config['type']}) via {actual_method}...")
            llm_client = create_llm_client(model_name, config, args.inference_method, args.num_gpus)
            
            # Process model
            model_processed_entries = process_single_model(
                model_name, llm_client, inferences_data, 
                args.max_workers, run_timestamp, output_base_dir, args.checkpoint_interval
            )
            
            # Stage 4: Restructure for Final Output
            print(f"--- Stage 4: Restructuring Data for Output ({model_name}) ---")
            original_item_map = {idx: {**item, "explanations_from_models": []} for idx, item in enumerate(inferences_data)}

            for processed_entry in model_processed_entries:
                original_idx = processed_entry["original_item_index"]
                target_item_in_map = original_item_map.get(original_idx)

                if target_item_in_map:
                    model_explanation_detail = {
                        "explanation_model_name": processed_entry["explanation_model_name"],
                        "inference": processed_entry["original_item_data"]["inference"],
                        "supporting_explanation": processed_entry["supporting_explanation"],
                        "opposing_explanation": processed_entry["opposing_explanation"],
                        "supporting_score_judge": processed_entry["supporting_score_judge"],
                        "opposing_score_judge": processed_entry["opposing_score_judge"],
                        "supporting_judge_raw_output": processed_entry["supporting_judge_raw_output"],
                        "opposing_judge_raw_output": processed_entry["opposing_judge_raw_output"],
                    }
                    target_item_in_map["explanations_from_models"].append(model_explanation_detail)

            final_outputs = [original_item_map[i] for i in sorted(original_item_map.keys())]
            
            # Stage 5: Add Bayes Scores
            print(f"--- Stage 5: Adding Bayes Scores ({model_name}) ---")
            final_outputs_with_bayes = add_bayes_scores(final_outputs)
            
            # Stage 6: Save Results and Plot
            print(f"--- Stage 6: Saving Results and Plotting ({model_name}) ---")
            model_output_dir = os.path.join(output_base_dir, model_name)
            results_dir = os.path.join(model_output_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            output_file = os.path.join(results_dir, f"{run_timestamp}.json")
            
            with open(output_file, 'w') as f:
                json.dump(final_outputs_with_bayes, f, indent=2)
                
            print(f"Saved {len(final_outputs_with_bayes)} processed items to {output_file}")

            # Generate plots
            if final_outputs_with_bayes:
                print(f"Generating score distribution plots for {model_name}...")
                plot_score_distributions(final_outputs_with_bayes, model_name, model_output_dir, run_timestamp)
            
            print(f"=== Completed processing for {model_name} ===")
            
        except Exception as e:
            print(f"[ERROR] Failed to process model '{model_name}': {e}")
            continue
        finally:
            # Best-effort GPU memory cleanup when switching models
            try:
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass

    print(f"\n=== Experiment Complete ===")
    print(f"All results saved to: {output_base_dir}")

if __name__ == "__main__":
    main()
