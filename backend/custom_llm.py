from typing import Any, List, Optional, Dict
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import google.generativeai as genai
import os

class CustomGeminiLLM(LLM):
    model_name: str = "gemini-flash-latest"
    google_api_key: str = None
    
    def __init__(self, model_name: str, google_api_key: str):
        super().__init__()
        self.model_name = model_name
        self.google_api_key = google_api_key
        genai.configure(api_key=google_api_key)

    @property
    def _llm_type(self) -> str:
        return "custom_gemini"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}
