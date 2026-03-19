from smolagents import CodeAgent, DuckDuckGoSearchTool, ApiModel, tool
import datetime
import requests
import pytz
import yaml
import uuid
import os
from bs4 import BeautifulSoup
from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI


@tool
def fetch_webpage(url: str) -> str:
    """Fetch and return the readable text content of a webpage.
    Args:
        url: The URL of the webpage to retrieve.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        text = soup.get_text(separator="\n", strip=True)
        return text[:5000]
    except Exception as e:
        return f"Error fetching webpage: {str(e)}"


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


@tool
def image_generation(prompt: str) -> str:
    """Generate an image from a text prompt using the Hugging Face Inference API.
    The tool sends the prompt to a text-to-image model and saves the
    resulting image locally, returning the file path.
    Args:
        prompt: A detailed text description of the image you want to generate.
    """
    import io
    from PIL import Image
    from huggingface_hub import InferenceClient

    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    OUTPUT_DIR = "generated_images"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODELS = [
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-3.5-large",
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "runwayml/stable-diffusion-v1-5",
    ]

    client = InferenceClient(token=HF_TOKEN)

    last_error = None
    for model_id in MODELS:
        try:
            image = client.text_to_image(
                prompt=prompt,
                model=model_id,
            )
            filename = f"{uuid.uuid4().hex[:12]}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            image.save(filepath)
            return f"Image saved to {filepath} (model: {model_id})"
        except Exception as e:
            last_error = e
            continue

    return f"Error generating image with all models. Last error: {str(last_error)}"


final_answer = FinalAnswerTool()
search_tool = DuckDuckGoSearchTool()

model = ApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    custom_role_conversions=None,
)

with open("prompts.yaml", "r") as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(
    model=model,
    tools=[
        final_answer,
        search_tool,
        fetch_webpage,
        get_current_time_in_timezone,
        image_generation,
    ],
    max_steps=8,
    verbosity_level=2,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates,
)

GradioUI(agent).launch()
