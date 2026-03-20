from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, tool
import datetime
import requests
import pytz
import uuid
import os
from bs4 import BeautifulSoup
from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI

###A Security Tool to audit headers using AI
@tool
def security_header_auditor(url: str) -> str:
    """
    This tool is a website security/recon/auditor used to identify headers of a website.
    
    Args:
        url: The URL of the webpage to retrieve.
    
    Returns:
        A Dictionary containing
        | Finding | Reason | Score |
    """
    try:
        response = requests.get(url, verify=False)
        headers = response.headers
        #All the Header Findings 
        headers_missing_finding = [
        "Strict-Transport-Security",
        "Content-Security-Policy",
        "X-Content-Type-Options",
        "X-Frame-Options",
        "Referrer-Policy",
        "Permissions-Policy",
        "X-Permitted-Cross-Domain-Policies",
        ]
        headers_present_finding = [
        "X-Powered-By",
        "X-AspNet-Version",
        "X-AspNetMvc-Version",
        "X-Generator",
        "X-Debug",
        "X-Debug-Token",
        "X-Debug-Token-Link",
        "X-Backend-Server",
        "X-Pingback",
        "SourceMap",
        "X-SourceMap",
        ]
        headers_deprecated = [
        "X-XSS-Protection",
        "Expect-CT",
        "Public-Key-Pins",
        "Public-Key-Pins-Report-Only",
        ]

        ###Storing of the findings
        findings = {
            "missing": {},
            "present": {},
            "deprecated": {}
        }
        ###The loop for all the missing headers
        for h in headers_missing_finding:
            if h not in response.headers:
                findings["missing"][h] = {
                    "status": "missing"
                }

        for h in headers_present_finding:
            if h in response.headers:
                findings["present"][h] = {
                    "status": "present"
                }

        for h in headers_deprecated:
            if h in response.headers:
                findings["deprecated"][h] = {
                    "status": "deprecated"
                }
        return str(findings)
        
    except Exception as e:
        return f"Error finding header security vulnerabilities: {str(e)}"
            
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
    """Fetch the current local time in a specified timezone.

    Args:
        timezone: A valid timezone string, e.g. 'America/New_York'.
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


@tool
def image_generation(prompt: str) -> str:
    """Generate an image from a text prompt using Hugging Face Inference API.

    Args:
        prompt: A detailed image description.
    """
    from huggingface_hub import InferenceClient

    hf_token = os.environ.get("HF_TOKEN", "")
    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)

    models = [
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-3.5-large",
        "runwayml/stable-diffusion-v1-5",
    ]

    client = InferenceClient(token=hf_token)

    last_error = None
    for model_id in models:
        try:
            image = client.text_to_image(prompt=prompt, model=model_id)
            filename = f"{uuid.uuid4().hex[:12]}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            return f"Image saved to {filepath} using model {model_id}"
        except Exception as e:
            last_error = e

    return f"Error generating image with all models. Last error: {str(last_error)}"


final_answer = FinalAnswerTool()
search_tool = DuckDuckGoSearchTool()

model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=os.environ.get("HF_TOKEN"),
)

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
    name="assistant",
    description="A helpful web-enabled assistant that can search, read webpages, check timezones, and generate images.",
)

GradioUI(agent).launch()
