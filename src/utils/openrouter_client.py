import os
import requests
import time
import threading
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": os.getenv("HTTP_REFERER", "https://www.clsp.jhu.edu/"),
    "X-Title": os.getenv("X_TITLE", "Social Reasoning NLI")
}

# Global, thread-safe rate limiter for OpenRouter API calls
def _parse_int_env(name, default):
    try:
        value = int(os.getenv(name, str(default)))
        return value if value >= 0 else default
    except Exception:
        return default

def _parse_float_env(name, default):
    try:
        value = float(os.getenv(name, str(default)))
        return value if value >= 0 else default
    except Exception:
        return default

class APICallRateLimiter:
    def __init__(self, calls_per_pause: int = 160, sleep_seconds: float = 5.0):
        self.calls_per_pause = int(calls_per_pause) if calls_per_pause is not None else 0
        self.sleep_seconds = float(sleep_seconds) if sleep_seconds is not None else 0.0
        self.counter = 0
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.pause_until = 0.0

    def before_call(self):
        # If disabled, still count but don't sleep
        if self.calls_per_pause <= 0 or self.sleep_seconds <= 0:
            with self.lock:
                self.counter += 1
            return

        should_sleep = False
        with self.lock:
            now = time.monotonic()
            while now < self.pause_until:
                remaining = self.pause_until - now
                # Wait until current pause has elapsed
                self.cond.wait(timeout=remaining)
                now = time.monotonic()

            self.counter += 1
            if self.counter % self.calls_per_pause == 0:
                self.pause_until = time.monotonic() + self.sleep_seconds
                should_sleep = True
                print(f"[RATE-LIMIT] Reached {self.counter} OpenRouter API calls. Pausing for {self.sleep_seconds} seconds...")

        if should_sleep:
            time.sleep(self.sleep_seconds)
            with self.lock:
                # Clear pause and wake up any waiting threads
                self.pause_until = 0.0
                self.cond.notify_all()

# Configure limiter via environment variables
_CALLS_PER_PAUSE = _parse_int_env("OPENROUTER_CALLS_PER_PAUSE", 160)
_PAUSE_SECONDS = _parse_float_env("OPENROUTER_PAUSE_SECONDS", 5.0)
_RATE_LIMITER = APICallRateLimiter(_CALLS_PER_PAUSE, _PAUSE_SECONDS)

def call_openrouter_chat_completion(model_id: str, system_prompt: str, user_prompt: str, max_retries: int = 3, timeout: int = 120):
    """Calls the OpenRouter API with chat completions."""
    if not OPENROUTER_API_KEY:
        print("[WARN] OPENROUTER_API_KEY not found in environment variables.")
        return {"content": "ERROR: OPENROUTER_API_KEY not found in environment variables.", "reasoning": None}

    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 5000
    }

    for attempt in range(max_retries):
        try:
            _RATE_LIMITER.before_call()
            response = requests.post(
                f"{OPENROUTER_API_BASE}/chat/completions",
                headers=HEADERS,
                json=data,
                timeout=timeout
            )
            response.raise_for_status()
            response_data = response.json()
            message = response_data.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")
            reasoning = message.get("reasoning")

            if reasoning:
                pass
            
            return {"content": content, "reasoning": reasoning}
        except requests.exceptions.RequestException as e:
            print(f"[WARN] OpenRouter API call (via util) failed (attempt {attempt + 1}/{max_retries}) for model {model_id}: {e}")
            if attempt == max_retries - 1:
                error_message = f"ERROR: OpenRouter API call (via util) failed after {max_retries} retries: {e}"
                return {"content": error_message, "reasoning": None}
            time.sleep(2 ** attempt)
        except (KeyError, IndexError, AttributeError) as e_json:
            response_text_snippet = response.text[:200] if response else "No response object"
            print(f"[WARN] OpenRouter API response structure error (via util) (attempt {attempt + 1}/{max_retries}) for model {model_id}: {e_json}. Response: {response_text_snippet}")
            if attempt == max_retries - 1:
                error_message = f"ERROR: OpenRouter API response structure error (via util) after {max_retries} retries: {e_json}"
                return {"content": error_message, "reasoning": None}
            time.sleep(2 ** attempt)

    final_error_message = f"ERROR: OpenRouter API call (via util) failed definitively after {max_retries} retries for model {model_id}."
    return {"content": final_error_message, "reasoning": None}

class OpenRouterLLM:
    def __init__(self, model_id: str, system_prompt: str = "You are a helpful assistant."):
        self.model_id = model_id
        self.system_prompt = system_prompt

    def invoke(self, messages: list) -> HumanMessage:
        if not messages or not isinstance(messages, list):
            raise ValueError("Input 'messages' must be a non-empty list.")

        user_prompt_message = messages[-1]
        if hasattr(user_prompt_message, 'content'):
            user_prompt = user_prompt_message.content
        elif isinstance(user_prompt_message, str):
            user_prompt = user_prompt_message
        else:
            raise ValueError("Last message in 'messages' must have a 'content' attribute or be a string.")
        
        current_system_prompt = self.system_prompt
        processed_messages_for_api = []

        if len(messages) > 1:
            print(f"[INFO] OpenRouterLLM received {len(messages)} messages. Constructing chat history.")

        response_data = call_openrouter_chat_completion(
            model_id=self.model_id,
            system_prompt=current_system_prompt, 
            user_prompt=user_prompt
        )
        return HumanMessage(content=response_data.get("content", ""))

def deepseek_r1_completion(prompt):
    """
    Invoke deepseek/deepseek-r1 on OpenRouter.
    Note: This function is specific to deepseek-r1 and uses a simplified message format.
    The generic `call_openrouter_chat_completion` or `OpenRouterLLM` class are preferred for other models.
    """
    data = {
        "model": "deepseek/deepseek-r1",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6,
        "max_tokens": 5000
    }
    try:
        _RATE_LIMITER.before_call()
        response = requests.post(
            f"{OPENROUTER_API_BASE}/chat/completions",
            headers=HEADERS,
            json=data,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[WARN] deepseek_r1_completion direct API call failed: {e}")
        return {"choices": [{"message": {"content": f"ERROR: deepseek_r1_completion failed: {e}"}}]}
    except Exception as ex_gen:
        print(f"[WARN] An unexpected error in deepseek_r1_completion: {ex_gen}")
        return {"choices": [{"message": {"content": f"ERROR: Unexpected error in deepseek_r1_completion: {ex_gen}"}}]}
