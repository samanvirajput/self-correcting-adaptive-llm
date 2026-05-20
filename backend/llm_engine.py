import requests
import json

class LLMEngine:
    def __init__(self, model_name: str = "phi3:mini", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = f"{host}/api/generate"
        print(f"✅ Using Ollama model '{self.model_name}' via {self.api_url}")

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Send a prompt to the local Ollama model and return the full text reply."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {"num_predict": max_new_tokens}
        }

        try:
            with requests.post(self.api_url, json=payload, stream=True, timeout=300) as response:
                response.raise_for_status()
                output = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            output += data["response"]
                return output.strip()

        except Exception as e:
            print(f"❌ Error generating response from Ollama: {e}")
            return "⚠️ Could not connect to Ollama. Make sure 'ollama serve' is running."
