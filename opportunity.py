import requests
import os
import json
import pandas as pd
import logging
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional
from ner_module import NER, ToolsManager
from ner_module.utils import NERResult
from rapidfuzz import fuzz, process

logging.getLogger("urllib3.connection").setLevel(logging.ERROR)  # Suppress urllib3 warnings

load_dotenv()

API_ID = os.getenv("API_ID")
API_PWD = os.getenv("API_PWD")
OPENAI_API = os.getenv("OPENAI_API")
CLIENTS = pd.read_excel("lista_clienti.xlsx")
CLIENTS_LIST = [client.strip().lower() for client in CLIENTS["ancl_ragione_sociale"].to_list()]

HOST = "https://opportunitystatwolf.opportunitycrm.it/API"

response = requests.post(
    f"{HOST}/auth/token",
    json={
        "apiId": API_ID,
        "apiPassword": API_PWD
    }
)

if response.status_code == 200:
    print("Authentication successful. Token received.")
    data = response.json()
    token = data.get("token")
    refresh_token = data.get("refreshToken")

HEADERS = {
    "authorization": f"Bearer {token}"
}

def get_activity_list() -> list[str]:
    response = requests.get(
        f"{HOST}/attivita/tipi",
        headers=HEADERS
    )
    activity_list = []

    if response.status_code == 200:
        print("Activity list retrieved successfully.")
        activities = response.json()
        for activity in activities:
            id = activity.get("id", "")
            codice = activity.get("codice", "")
            descrizione = activity.get("descrizione", "")
            todo = activity.get("todo", False)
            act = f"id: {id}, codice: {codice}, descrizione: {descrizione}, todo: {todo}"
            activity_list.append(act)
    else:
        print("Failed to retrieve activity list.")

    return activity_list

def get_user_id() -> int:
    return 38 # ONLY FOR TESTING PURPOSES -> THIS ID SHOULD BE DYNAMICALLY RETRIEVED FROM THE METADATA
        
class Activity(BaseModel):
    id_attivita: int = Field(..., description="ID dell'attività individuata.")
    is_todo: bool = Field(..., description="Indica se l'attività è un todo (da fare) o meno. E' strettamente collegato all'attività. Se True indica un'attività in corso d'opera, False un'attività completata.")
    codice_todotype: Optional[str] = Field(default=None, description="Codice del tipo di attività, presente vicino alla descrizione dell'attività.")
    id_stato: int = Field(..., description="ID dello stato dell'attività. 1 = Aperto, 2 = Chiuso, 3 = In corso.")
    data_registrazione: datetime = Field(default=datetime.now(), description="Data di registrazione dell'attività in formato ISO 8601, che corrisponde alla data attuale.")
    data_inizio: datetime = Field(..., description="Data di inizio dell'attività in formato ISO 8601 che l'utente deve definire.")
    data_fine: datetime = Field(..., description="Data di fine dell'attività in formato ISO 8601 che l'utente deve definire.")
    data_esecuzione: Optional[datetime] = Field(default=None, description="Data di esecuzione dell'attività in formato ISO 8601. Se non specificata, viene impostata a None, viene cambiata quando viene effettivamente eseguito il todo.")
    giornata_intera: bool = Field(default=False, description="Indica se l'attività è di tutta la giornata.")
    oggetto: Optional[str] = Field(default="", description="Oggetto dell'attività.")
    testo: Optional[str] = Field(default="", description="Testo dell'attività. Completa descrizione delle informazioni rilevate nel testo.")
    note: Optional[str] = Field(default="", description="Note aggiuntive sull'attività. Note aggiuntive utili, come date, luoghi o altro.")
    destinazione: Optional[str] = Field(default=None, description="Destinazione (indirizzo, via, città) specificato per l'attività.")
    id_utente: int = Field(default=get_user_id(), description="ID dell'utente che ha creato l'attività.")
    id_calendario: Optional[int] = Field(default=None, description="ID del calendario dell'utente dove inserire l'attività.")
    cliente: str = Field(default="", description="Cliente associato all'attività")
    id_cliente: Optional[int] = Field(default=None, description="ID del cliente associato al contatto. Azienda in cui lavora il contatto se presente.")
    contatto: Optional[str] = Field(default=None, description="Nome completo del contatto associato all'attività.")
    id_contatto: Optional[int] = Field(default=None, description="ID del contatto associato all'attività.")
    id_att_parent: Optional[int] = Field(default=None, description="ID dell'attività padre, se presente. Utilizzato per attività ricorrenti o correlate.")

    @model_validator(mode="before")
    def set_some_data(cls, values):
        print("Before model validation.")
        """
        if values.get("id_attivita") in [1, 2, 3]:
            if values.get("data_registrazione") is None:
                values["data_registrazione"] = datetime.now()
            if values.get("data_inizio") is None:
                values["data_inizio"] = values["data_registrazione"] + timedelta(days=1) # Default to one day after registration date if not specified
            if values.get("data_fine") is None or values.get("data_inizio") >= values.get("data_fine"):
                values["data_fine"] = values["data_inizio"] + timedelta(hours=1) # Default to one hour after start date if not specified
        """
        return values

    @model_validator(mode="after")
    def set_data_fine_if_missing(self):
        print("After model validation.")
        return self

# text = "Dopo un confronto con Siziano, l'agente ha deciso di procedere con l'attività: invio offerta. Il cliente Siziano ha espresso potenziale interesse, il colloquio è fissato per domani alle 11 di mattina."
# text = "Dopo un confronto con Siziano, l'agente ha deciso di procedere con l'attività: invio offerta. Il cliente Siziano ha espresso potenziale interesse."
# text = "Domani invia un'offerta a Siziano e apri ad un eventuale incontro con Statwolf."
text = "Dopo un confronto con Siziano siamo pronti a procedere con l'invio dell'offerta."

print("\n\nINPUT TEXT\n", text, "\n")
# Using agents with tools

# Initialize tools manager
tools_manager = ToolsManager(host=HOST, bearer_token=token)

# Add any necessary tools to the manager

# Tool to retrieve the client ID from the client name
def make_get_client_id_tool(host: str, token: str):
    headers = {"Authorization": f"Bearer {token}"}
    def get_client_id(client_name):
        """
        Retrieve the client ID given a client name using fuzzy matching.

        :param client_name: Name of the client (partial or full)
        :return: The ID of the client if found, otherwise None.
        """
        THR = 50
        clients = pd.read_excel("lista_clienti.xlsx")
        clients_list = [client.strip().lower() for client in clients["ancl_ragione_sociale"].to_list()]

        client_name = client_name.strip().lower()
        match = process.extractOne(client_name, clients_list, scorer=fuzz.token_sort_ratio)
        best_match, score = match[0], match[1]
        client = best_match if score >= THR else None
        if client:
            params = {
                "RagSoc": client
            }
            response = requests.get(
                f"{host}/clienti",
                headers=headers,
                params=params
            )

            if response.status_code == 200:
                client = response.json()[0]
                return client.get("id", None)
            else:
                return None
        else:
            return None
        
    return get_client_id

def make_get_todo_from_activity(host: str, token: str):
    headers = {"Authorization": f"Bearer {token}"}
    def get_todo_from_activity(activity_id: int):
        """
        Get if the activity is a todo or not.

        :param activity_id: The ID of the activity.
        :return: True if the activity is a todo, False otherwise.
        """
        response = requests.get(
            f"{host}/attivita/tipi",
            headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            for activity in data:
                if activity.get("id") == activity_id:
                    return activity.get("todo", False)
        
        return False
    
    return get_todo_from_activity

def make_get_calendar_id_tool(host: str, token: str):
    headers = {"Authorization": f"Bearer {token}"}
    def get_calendar_id(user_id, default = True):
        """
        Retrieve the calendar ID using the user_id.

        :param user_id: ID of the user
        :param default: If True, return the default calendar ID for the user; otherwise, return the secondary calendar ID.
        :return: The calendar ID for the user.
        """
        params = {"idUtente": user_id}
        response = requests.get(
            f"{host}/utenti/calendariUtente",
            headers=headers,
            params=params
        )
        user_calendars = response.json()
        for calendar in user_calendars:
            if calendar.get("default", False) == default:
                return calendar.get("idCalendario")

    return get_calendar_id
    
def make_get_todotype(host: str, token: str):
    headers = {"Authorization": f"Bearer {token}"}
    def get_todotype(id_activity: int):
        """
        Retrieve the TodoType using the activity ID.
        The TodoType is the code relative to the activity type.

        :param id_activity: The ID of the activity.
        :return: The TodoType if found, otherwise None.
        """
        response = requests.get(
            f"{host}/attivita/tipi",
            headers=headers
        )

        if response.status_code == 200:
            activities = response.json()
            for activity in activities:
                if activity.get("id") == id_activity:
                    return activity.get("codice", None)
            return None
        else:
            return None

    return get_todotype


def make_get_all_users(host: str, token: str):
    headers = {"Authorization": f"Bearer {token}"}
    def get_all_users():
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

    return get_all_users

def make_get_current_date():
    def get_current_date():
        """
        Get the current day, month, year and time.
        
        :return: The current day, month, year and time as a formatted string.
        """
        return datetime.now().strftime("%A %d %B %Y %H %M %S")
    return get_current_date

tools_manager.add_fn_to_tools(make_get_all_users(tools_manager.host, tools_manager.token))
tools_manager.add_fn_to_tools(make_get_client_id_tool(tools_manager.host, tools_manager.token))
tools_manager.add_fn_to_tools(make_get_calendar_id_tool(tools_manager.host, tools_manager.token))
tools_manager.add_fn_to_tools(make_get_todo_from_activity(tools_manager.host, tools_manager.token))
tools_manager.add_fn_to_tools(make_get_todotype(tools_manager.host, tools_manager.token))
tools_manager.add_fn_to_tools(make_get_current_date())

print(tools_manager)

ner = NER(api_key=OPENAI_API)

activity_list = get_activity_list()
activities = ner.detect_multiactivities(text=text)

print("Number of activities detected: ", len(activities))
for i, act in enumerate(activities, start=1):
    print(f"Activity {i}: {act}")

for text in activities:
    print("PROCESSING TEXT: ", text)
    print("-" * 80)
    complete = False
    additional_parameters = {}
    while not complete:
        result = ner.chat(
            text=text,
            tasks=activity_list,
            base_model=Activity,
            tools_manager=tools_manager,
            ask_missing_params_with_llm=True,
            user_inputs=additional_parameters if additional_parameters else None,
            return_raw_result=True
        )
        additional_parameters.clear()
        if isinstance(result, NERResult) and result.success:
            complete = result.success
        elif isinstance(result, NERResult) and not result.success and result.status == "missing_parameter":
            # The parameters can be requested in different ways to the user
            for item in result.questions:
                item = json.loads(item) if isinstance(item, str) else item
                param, question = item.values()
                answer = input(question)
                additional_parameters[param] = answer
        else:
            raise RuntimeError("NER processing failed.")

    print("-" * 80)
    print("FINAL RESULT: ")
    result_data = json.loads(result.data)
    for key, value in result_data.items():
        print(f"{key}: {value}" if value is not None else f"{key}: None")