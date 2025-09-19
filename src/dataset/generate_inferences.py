from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pathlib import Path

if __package__ in (None, ""):
    current_dir = Path(__file__).resolve().parent
    src_root = current_dir.parent
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from dotenv import load_dotenv
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from prompts.inference_cot import generate_social_nli_cot
from prompts.inference_no_cot import generate_social_nli_no_cot
from utils.openrouter_client import OpenRouterLLM

load_dotenv()

_enable_langchain_cache = os.getenv("SONLI_ENABLE_LANGCHAIN_CACHE", "0").lower() in {
    "1",
    "true",
    "yes",
}
if _enable_langchain_cache:
    try:
        set_llm_cache(SQLiteCache(database_path=".langchain_inference_cache.db"))
        print("[INFO] LangChain SQLite cache enabled.")
    except Exception as cache_exc:  # noqa: BLE001
        print(f"[WARN] Failed to initialize LangChain SQLite cache: {cache_exc}")
else:
    print("[INFO] LangChain cache disabled (set SONLI_ENABLE_LANGCHAIN_CACHE=1 to enable).")

DEFAULT_INPUT_PATH = (
    "datasets/socialnli_sources/curated/"
    "friendsqa_dialogues_qa_sarcasm_irony_only.json"
)
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_RUN_NAME = "socialnli_inference_regen"

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
GENERIC_BLOCK_PATTERN = re.compile(r"```\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
FIRST_OBJECT_PATTERN = re.compile(r"(\{[\s\S]*?\})")


@dataclass
class InferenceRequest:
    model_name: str
    provider: str  # "openai" or "openrouter"
    prompt_variant: str  # "cot" or "no_cot"
    prompt_builder: Any
    temperature: float = 0.7
    max_retries: int = 3
    retry_backoff: float = 2.0


@dataclass
class InferenceOutput:
    model_name: str
    prompt_variant: str
    inferences: List[str]
    raw_json: Dict[str, Any]
    reasoning: Optional[str]
    raw_response: str
    latency_seconds: float
    error: Optional[str] = None

    def to_serializable(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


class LLMInvoker:
    """Wrapper that lazily instantiates the appropriate model client."""

    def __init__(self, request: InferenceRequest, openai_api_key: Optional[str], openrouter_system_prompt: str):
        self.request = request
        self.openai_api_key = openai_api_key
        self.openrouter_system_prompt = openrouter_system_prompt
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if self.request.provider == "openai":
                if not self.openai_api_key:
                    raise RuntimeError(
                        "OPENAI_API_KEY not set but provider 'openai' was requested."
                    )
                self._client = ChatOpenAI(
                    model=self.request.model_name,
                    temperature=self.request.temperature,
                    api_key=self.openai_api_key,
                    max_tokens=4096,
                )
            elif self.request.provider == "openrouter":
                self._client = OpenRouterLLM(
                    model_id=self.request.model_name,
                    system_prompt=self.openrouter_system_prompt,
                )
            else:
                raise ValueError(f"Unsupported provider '{self.request.provider}'.")
        return self._client

    def invoke(self, prompt: str) -> str:
        message = HumanMessage(content=prompt)
        response = self.client.invoke([message])
        if hasattr(response, "content"):
            return response.content
        return str(response)


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def join_dialogue(dialogue: Iterable[str]) -> str:
    return "\n".join(dialogue)


def parse_reasoning(text: str) -> Optional[str]:
    match = THINK_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def _try_parse_json(candidate: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def extract_json_block(response_text: str) -> Optional[Dict[str, Any]]:
    for pattern in (JSON_BLOCK_PATTERN, GENERIC_BLOCK_PATTERN):
        match = pattern.search(response_text)
        if match:
            parsed = _try_parse_json(match.group(1))
            if parsed:
                return parsed
    
    cleaned = response_text.strip()
    parsed = _try_parse_json(cleaned)
    if parsed:
        return parsed

    # Fallback: grab first object-looking substring.
    match = FIRST_OBJECT_PATTERN.search(response_text)
    if match:
        return _try_parse_json(match.group(1))
    return None


def normalize_inference_list(data: Dict[str, Any], expected: int = 5) -> List[str]:
    results: List[str] = []
    for idx in range(1, expected + 1):
        item = data.get(str(idx))
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                results.append(cleaned)
    return results


def generate_inferences_for_question(
    invoker: LLMInvoker,
    prompt_builder,
    question: str,
    scene_dialogue: str,
    expected_count: int = 5,
) -> Tuple[List[str], Dict[str, Any], Optional[str], str, float, Optional[str]]:
    prompt = prompt_builder(question, scene_dialogue)
    attempts = 0
    error_message: Optional[str] = None
    start_time = time.perf_counter()
    raw_response = ""
    json_payload: Optional[Dict[str, Any]] = None

    while attempts < invoker.request.max_retries:
        attempts += 1
        try:
            raw_response = invoker.invoke(prompt)
            json_payload = extract_json_block(raw_response)
            if json_payload:
                inferences = normalize_inference_list(json_payload, expected=expected_count)
                if len(inferences) == expected_count:
                    error_message = None
                    break
            error_message = (
                f"JSON parse failed or missing {expected_count} keys on attempt {attempts}."
            )
        except Exception as exc:  # noqa: BLE001
            error_message = f"LLM invocation failed on attempt {attempts}: {exc}"
        time.sleep(invoker.request.retry_backoff ** (attempts - 1))

    latency = time.perf_counter() - start_time

    if not json_payload:
        json_payload = {}
    inferences = normalize_inference_list(json_payload, expected=expected_count)
    reasoning = parse_reasoning(raw_response)
    return inferences, json_payload, reasoning, raw_response, latency, error_message


def ensure_output_dir(base_dir: str, run_name: str) -> Tuple[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, "socialnli_inferences.json")
    return run_dir, output_path


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SocialNLI-style inferences using gpt-4o (CoT) and gpt-3.5 (no-CoT)."
    )
    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Path to FriendsQA-derived input JSON (default: curated sarcasm/irony subset).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory for writing outputs (default: outputs/).",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="Label for the run directory that will be created under output-dir.",
    )
    parser.add_argument(
        "--cot-model",
        default="gpt-4o",
        help="Model name for CoT generation (default: gpt-4o).",
    )
    parser.add_argument(
        "--cot-provider",
        choices=["openai", "openrouter"],
        default="openai",
        help="Provider for CoT model (default: openai).",
    )
    parser.add_argument(
        "--no-cot-model",
        default="gpt-3.5-turbo",
        help="Model name for non-CoT generation (default: gpt-3.5-turbo).",
    )
    parser.add_argument(
        "--no-cot-provider",
        choices=["openai", "openrouter"],
        default="openai",
        help="Provider for non-CoT model (default: openai).",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of question-answer pairs to process.",
    )
    parser.add_argument(
        "-s",
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay (in seconds) between API calls to avoid rate limits.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw LLM responses in the output JSON (default: store them but flag for awareness).",
    )
    return parser


def main(args: Optional[List[str]] = None) -> None:
    parser = build_cli()
    parsed = parser.parse_args(args=args)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_system_prompt = os.getenv(
        "OPENROUTER_SYSTEM_PROMPT", "You are a helpful assistant for SocialNLI data generation."
    )

    dataset = load_dataset(parsed.input)

    qa_records: List[Dict[str, Any]] = []
    for entry in dataset:
        dialogue_list = entry.get("dialogue")
        if not dialogue_list:
            continue
        dialogue_text = join_dialogue(dialogue_list)
        qas = entry.get("qas", [])
        for qa in qas:
            qa_records.append(
                {
                    "dialogue_id": entry.get("title"),
                    "question_id": qa.get("id"),
                    "question": qa.get("question"),
                    "answers": qa.get("answers", []),
                    "dialogue_text": dialogue_text,
                    "dialogue": dialogue_list,
                    "metadata": {
                        "contains_sarcasm": entry.get("contains_sarcasm"),
                        "contains_irony": entry.get("contains_irony"),
                    },
                }
            )

    if parsed.limit is not None:
        qa_records = qa_records[: parsed.limit]

    run_dir, output_path = ensure_output_dir(parsed.output_dir, parsed.run_name)

    cot_request = InferenceRequest(
        model_name=parsed.cot_model,
        provider=parsed.cot_provider,
        prompt_variant="cot",
        prompt_builder=generate_social_nli_cot,
    )
    no_cot_request = InferenceRequest(
        model_name=parsed.no_cot_model,
        provider=parsed.no_cot_provider,
        prompt_variant="no_cot",
        prompt_builder=generate_social_nli_no_cot,
    )

    cot_invoker = LLMInvoker(cot_request, openai_api_key=openai_api_key, openrouter_system_prompt=openrouter_system_prompt)
    no_cot_invoker = LLMInvoker(no_cot_request, openai_api_key=openai_api_key, openrouter_system_prompt=openrouter_system_prompt)

    outputs: List[Dict[str, Any]] = []
    total = len(qa_records)
    print(f"Processing {total} question-answer pairs...")

    for idx, qa in enumerate(qa_records, start=1):
        info_prefix = f"[{idx}/{total}]"
        question = qa.get("question") or ""
        dialogue_text = qa.get("dialogue_text") or ""

        print(f"{info_prefix} CoT inferences via {cot_request.model_name}...")
        cot_inferences, cot_json, cot_reasoning, cot_raw, cot_latency, cot_internal_error = generate_inferences_for_question(
            cot_invoker,
            cot_request.prompt_builder,
            question,
            dialogue_text,
        )
        cot_error = cot_internal_error
        if len(cot_inferences) < 5:
            shortage_msg = (
                f"Expected 5 CoT inferences but received {len(cot_inferences)} from {cot_request.model_name}."
            )
            cot_error = f"{cot_error}; {shortage_msg}" if cot_error else shortage_msg
            print(f"  [WARN] {cot_error}")

        if parsed.sleep:
            time.sleep(parsed.sleep)

        print(f"{info_prefix} Non-CoT inferences via {no_cot_request.model_name}...")
        no_cot_inferences, no_cot_json, no_cot_reasoning, no_cot_raw, no_cot_latency, no_cot_internal_error = generate_inferences_for_question(
            no_cot_invoker,
            no_cot_request.prompt_builder,
            question,
            dialogue_text,
        )
        no_cot_error = no_cot_internal_error
        if len(no_cot_inferences) < 5:
            shortage_msg = (
                f"Expected 5 non-CoT inferences but received {len(no_cot_inferences)} from {no_cot_request.model_name}."
            )
            no_cot_error = f"{no_cot_error}; {shortage_msg}" if no_cot_error else shortage_msg
            print(f"  [WARN] {no_cot_error}")

        result_entry: Dict[str, Any] = {
            "dialogue_id": qa.get("dialogue_id"),
            "question_id": qa.get("question_id"),
            "question": question,
            "answers": qa.get("answers", []),
            "dialogue": qa.get("dialogue", []),
            "metadata": qa.get("metadata", {}),
            "inference_sets": {
                "gpt-4o": InferenceOutput(
                    model_name=cot_request.model_name,
                    prompt_variant=cot_request.prompt_variant,
                    inferences=cot_inferences,
                    raw_json=cot_json,
                    reasoning=cot_reasoning,
                    raw_response=cot_raw if parsed.include_raw else "",
                    latency_seconds=cot_latency,
                    error=cot_error,
                ).to_serializable(),
                "gpt-3.5": InferenceOutput(
                    model_name=no_cot_request.model_name,
                    prompt_variant=no_cot_request.prompt_variant,
                    inferences=no_cot_inferences,
                    raw_json=no_cot_json,
                    reasoning=no_cot_reasoning,
                    raw_response=no_cot_raw if parsed.include_raw else "",
                    latency_seconds=no_cot_latency,
                    error=no_cot_error,
                ).to_serializable(),
            },
        }

        outputs.append(result_entry)

        if parsed.sleep:
            time.sleep(parsed.sleep)

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "total_pairs": total,
        "input_path": os.path.abspath(parsed.input),
        "cot_model": {
            "name": cot_request.model_name,
            "provider": cot_request.provider,
        },
        "no_cot_model": {
            "name": no_cot_request.model_name,
            "provider": no_cot_request.provider,
        },
    }

    final_payload = {
        "metadata": metadata,
        "data": outputs,
    }

    with open(output_path, "w") as f:
        json.dump(final_payload, f, indent=2)

    print(f"Saved inference outputs to {output_path}")


if __name__ == "__main__":
    main()
