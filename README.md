# Ner Module

The NER Module provides a set of tools and utilities for Named Entity Recognition tasks. It is designed to be easy to use and integrate into existing workflows.

To install all the requirements to run this module we suggest to create a new environment:
- Conda:
    ```bash
    # Create a conda environment called "nerenv" (choose the name that you prefer)
    conda create -n nerenv python=3.10
    conda activate nerenv
    ```
- Venv:
    ```bash
    # Create a venv in the current directory
    python -m venv nerenv

    # Activate it (Linux/macOS)
    source nerenv/bin/activate

    # Activate it (Windows)
    .\nerenv\Scripts\activate
    ```

The install the dependencies inside the virtual environment with:
```bash
pip install -r requirements.txt
```

We provided the ```Read-the-docs``` html page to navigate across the module components.

## Example of usage

```python
from ner_module import NER, ToolsManager
from ner_module.utils import NERResult
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional

# For the Named Entity Recognition is possible to use different providers like: OpenAI, HuggingFace, Anthropic and Ollama. For Ollama you don't need an api key.
ner = NER(provider="openai", api_key="YOUR_API_KEY")

text = "Riunione con il cliente Mario Rossi il 15 Settembre 2025 e ricordami del viaggio aziendale del 27 Ottobre 2025."

# We suggest to create a Pydantic BaseModel child class in order to highlight the entities
# In this way the LLM will respond with a structured JSON output more consistently.
class Entity(BaseModel):
    attività: str = Field(..., description="L'attività da svolgere dall'utente.")
    data: Optional[datetime] = Field(default=None, description="La data prevista per lo svolgimento dell'attività.")
    cliente: Optional[str] = Field(default=None, description="Nome del cliente se presente.")

# Is possible to use different tools
# With tools we mean methods that the LLM can use to retrieve some important information.
# Using host and bearer token parameter is possible to give the LLM access to your APIs (optional parameters).
# IMPORTANT: you need to create this functions, the LLM doesn't know how to do this API calls.
tools_manager = ToolsManager(host="YOUR_HOST", bearer_token="YOUR_BEARER_TOKEN")

# This is a list of all the possible activities that can happen
possible_activities = [
    "Viaggio Aziendale",
    "Riunione",
    "Incontro diretto",
    "Pranzo Aziendale"
]

# This is a method to retrieve the current date and time. This method is constructed inside another one.
# This is the structure needed to create a tool.
# IMPORTANT: there must be the docstring, try to make these method description more complete than possible. The LLM doesn't read the code, it read the docstring and it is needed, it calls the method.
def make_get_current_date():
    def get_current_date():
        """
        Get the current day, month, year and time.

        Examples: 
        - If Today is Monday 25 December 2023 14 30 00, tomorrow is Tuesday 26 December 2023 14 30 00.
        - If Today is Tuesday 26 December 2023 14 30 00, from eight days is Wednesday 03 January 2024 14 30 00.

        :return: The current day, month, year and time as a formatted string.
        """
        return f"Today Date Information: {datetime.now().strftime('%A %d %B %Y %H %M %S')}"
    return get_current_date

def make_get_users(host: str, token: str):
    headers = {"Authorization": f"Bearer {token}"}
    def get_users():
        """
        Return the list of all the users.
        This function should be called ONLY when an activity is assigned to another person or entity (e.g. "assign this activity to John Doe/the marketing team").

        :return: A list of all users.
        """
        response = requests.get(
            f"{host}/utenti",
            headers=headers
        )
        if response.status_code == 200:
            users = response.json()
            users_infos = []
            for user in users:
                infos = f"ID: {user.get('id', '')}, Fullname: {user.get('fullname', '')}, Name: {user.get('nome', '')}, Cognome: {user.get('cognome', '')}, Email: {user.get('email', '')}, Cellphone: {user.get('cellulare', '')}"
                users_infos.append(infos)
            return users_infos
        else:
            return []

    return get_users


tools_manager.add_fn_to_tools(make_get_current_date())
tools_manager.add_fn_to_tools(make_get_users(host=host, token=token))

# Method to divide different activities in text
activities = ner.detect_multiactivities(text=text)

print("Detected activities:", activities)

for activity_text in activities:
    # Chat is the main method of the module, is possible to pass different parameters.
    # Is also possible to ask for missing parameter with the use of LLM or not.
    complete = False
    additional_parameters = {}
    while not complete:
        result = ner.chat(
            text=activity_text,
            tasks=possible_activities,
            base_model=Entity,  # Optional: can be a Pydantic model or not
            tools_manager=tools_manager, # Optional: can be a tool manager or not
            ask_missing_params_with_llm=True, # True -> LLM asks for parameters, False -> Asks without LLM
            return_raw_result=True # True -> Return a NERResult object (see ner_module.utils), False -> return a dict
        )
        additional_parameters.clear()
        if isinstance(result, NERResult) and result.success:
            complete = result.success
        elif isinstance(result, NERResult) and not result.success and result.status == "missing_parameter":
            # The parameters can be requested in different ways to the user (your choice)
            for item in result.questions:
                item = json.loads(item) if isinstance(item, str) else item
                param, question = item.values()
                answer = input(question)
                additional_parameters[param] = answer
        else:
            raise RuntimeError("NER processing failed.")
    print("\nProcessed activity result:\n", result.data)
```

