import os
import requests


def _ollama_running(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


class OllamaClient:
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.backend = "ollama"

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        import json
        payload = {"model": self.model, "prompt": prompt, "options": {"num_predict": max_tokens}}
        try:
            with requests.post(self.api_url, json=payload, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                out = ""
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            out += data["response"]
                return out.strip()
        except Exception as e:
            return f"[Ollama error: {e}]"

    def info(self) -> dict:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2).json()
            models = [m["name"] for m in r.get("models", [])]
            return {"backend": "ollama", "model": self.model, "available_models": models}
        except Exception:
            return {"backend": "ollama", "model": self.model}


class LlamaCppClient:
    def __init__(self, model_path: str):
        from llama_cpp import Llama
        self.model_path = model_path
        self.llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
        self.backend = "llama-cpp"

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        out = self.llm(prompt, max_tokens=max_tokens, echo=False)
        return out["choices"][0]["text"].strip()

    def info(self) -> dict:
        return {"backend": "llama-cpp", "model": self.model_path, "quantization": "GGUF"}


def load_model():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    model_path = os.getenv("MODEL_PATH", "")

    if _ollama_running(base_url):
        print(f"[loader] Ollama detected — loading '{ollama_model}'")
        return OllamaClient(model=ollama_model, base_url=base_url)

    if model_path and os.path.exists(model_path):
        print(f"[loader] Ollama not found — falling back to llama-cpp: {model_path}")
        return LlamaCppClient(model_path=model_path)

    print("[loader] WARNING: no model backend found. Returning stub.")
    return _StubClient()


class _StubClient:
    backend = "stub"

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        return "[No model loaded — start Ollama with 'ollama serve' or set MODEL_PATH in .env]"

    def info(self) -> dict:
        return {"backend": "stub", "model": "none"}


def get_model_info(client) -> dict:
    return client.info()
