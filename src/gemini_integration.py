import json
import textwrap
from typing import Dict, Any
import pandas as pd

# Import from our own project modules
from .config import GeminiConfig

# ==============================================================================
# Gemini API Integration for Moderation
# ==============================================================================

class GeminiModerator:
    """
    Handles interaction with the Gemini API for structured content moderation.

    This class is configured via a GeminiConfig object and encapsulates the logic
    for creating prompts, defining the expected JSON output schema, and simulating
    the API call.
    """

    def __init__(self, cfg: GeminiConfig):
        """
        Initializes the GeminiModerator with a configuration object.

        Args:
            cfg (GeminiConfig): A dataclass containing settings for the Gemini API,
                                such as the model name.
        """
        self.cfg = cfg
        self.model_name = cfg.model_name

    def get_moderation_schema(self) -> Dict[str, Any]:
        """
        Defines the JSON schema for the structured output from the Gemini API.
        ... (rest of the method is unchanged) ...
        """
        return {
            "type": "object",
            "properties": {
                "should_delete": {
                    "type": "boolean",
                    "description": "The final moderation decision (True to delete, False to keep)."
                },
                # ----- AFTER -----
                "confidence": {
                    "type": "number",
                    "description": "The model's confidence in its decision."
                },
                "reason": {
                    "type": "string",
                    "enum": ["spam", "inappropriate_content", "low_quality", "off_topic", "acceptable"],
                    "description": "The primary reason for the moderation decision."
                },
                "explanation": {
                    "type": "string",
                    "description": "A concise, one-sentence explanation for the decision."
                },
            },
            "required": ["should_delete", "confidence", "reason", "explanation"],
        }

    def build_moderation_prompt(self, post_data: pd.Series) -> str:
        """
        Constructs a detailed prompt for the Gemini API based on post data.
        ... (rest of the method is unchanged) ...
        """
        title = post_data.get("title", "N/A")
        score = post_data.get("score", 0)
        author = post_data.get("author", "unknown")
        awards = post_data.get("total_awards_received", 0)

        prompt = f"""
You are an expert content moderator for the subreddit r/dataisbeautiful. Your task is to analyze the following post's metadata and decide if it should be removed.

Provide your decision as a JSON object that strictly follows the provided schema.

**Post Metadata:**
- **Title:** "{title}"
- **Score:** {score}
- **Author:** {author}
- **Total Awards:** {awards}

**Moderation Guidelines:**
1.  **Low Quality/Effort:** Remove posts with vague titles or very low scores.
2.  **Off-Topic:** Ensure the post is related to data visualization.
3.  **Spam/Inappropriate:** Immediately remove any posts that are clearly spam or contain inappropriate content.
4.  **Community Reception:** High scores and awards are strong signals that the content is acceptable.

Return only the valid JSON object, with no additional text or explanations outside of the JSON structure.
        """.strip()
        return prompt

    def get_sdk_usage_example(self, prompt: str) -> str:
        """
        Generates a string of example Python code demonstrating how to use the Gemini SDK.
        ... (rest of the method is unchanged) ...
        """
        schema_str = json.dumps(self.get_moderation_schema())
        
        return textwrap.dedent(f"""
        # PSEUDOCODE: This demonstrates how to call the Gemini API with the Python SDK.
        # Ensure you have the 'google-generativeai' library installed and configured.
        
        import google.generativeai as genai
        
        # genai.configure(api_key="YOUR_API_KEY")

        generation_config = {{
            "response_mime_type": "application/json",
            "response_schema": {schema_str}
        }}

        model = genai.GenerativeModel(
            model_name="{self.cfg.model_name}",
            generation_config=generation_config
        )

        prompt_to_send = \"\"\"{prompt}\"\"\"
        # response = model.generate_content(prompt_to_send)
        # print(response.text)
        """)

# ==============================================================================
# Main execution block for demonstration
# ==============================================================================

if __name__ == '__main__':
    print("🚀 Running Gemini Integration Module Demonstration 🚀")
    
    # To run this file directly, we first need to create a config object
    demo_config = GeminiConfig(model_name="gemini-1.5-flash-latest")
    
    gemini_moderator = GeminiModerator(cfg=demo_config)
    print(f"Initialized moderator with model: {gemini_moderator.cfg.model_name}")

    sample_post = pd.Series({
        "title": "[OC] The Rise and Fall of Popular Programming Languages (2004-2024)",
        "score": 2580, "author": "dataviz_guru", "total_awards_received": 12
    })
    print("\nSample post for testing:\n" + sample_post.to_string())
    
    generated_prompt = gemini_moderator.build_moderation_prompt(sample_post)
    print("\nGenerated Prompt for the Gemini API:\n" + generated_prompt)

    sdk_example_code = gemini_moderator.get_sdk_usage_example(generated_prompt)
    print("\nExample Python SDK Usage (Pseudocode):\n" + sdk_example_code)
    
    print("✅ Demonstration complete.")
