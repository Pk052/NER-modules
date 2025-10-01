import json
import logging
import warnings
import pycountry
from enum import Enum
from pydantic import BaseModel, ValidationError, SecretStr
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.exceptions import LangChainException
from langchain.agents import initialize_agent, AgentType
from langdetect import detect
from openai import OpenAIError
from typing import List, Type, Any, Dict
from .config import PROVIDER, MODEL_NAME, TEMPERATURE
from .prompts import SYS_PROMPT, PROMPT, PROMPT_PARAMS_RECOVERY, PROMPT_PARAMETER_ASKING, PROMPT_BASE_MODEL_SELECTOR, PROMPT_TYPE_CASTING, PROMPT_MULTIACTIVITY
from .tools import ToolsManager
from .utils import NERMissingParameters, NERError, NERResult, ProcessingMode, BaseModelSelector, MultiActivity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)

class NER:
    """
    This is the Named Entity Recognition (NER) class.

    In this class, we define the methods and attributes necessary for performing NER tasks using a language model.
    The prompts defined in the prompts module are used to guide the model's behavior.
    In addition, utilizing a tool manager allows for better organization and management of the various tools that can be used during the NER process.

    The tools manager is responsible for registering, unregistering, and retrieving tools as needed throughout the NER workflow.
    It is possible to change the parameters of the NER class after it has been initialized.

    Is also possible to change the parameters of the model being used for NER tasks using the `set_model_params` method.
    To retrieve the model parameters, you can use the `get_model_params` method.

    The `chat` method is responsible for processing user input and generating a structured response based on the defined NER tasks and prompts.
    The base_model is used to define the structure of the expected output. In this way, it ensures that the model's responses adhere to the specified format.
    The tasks defined the entities to be recognized. These tasks guide the model in understanding what specific information needs to be extracted from the user input.
    If a tool_manager is provided, it can be used to access additional tools and resources that may be helpful during the NER process. In this case, the tool_manager
    can be utilized to enhance the capabilities of the NER class by providing access to external tools and APIs that can assist in the recognition and extraction of entities.
    The extra_info parameter can be used to pass any additional context or information that may be relevant to the NER process. It is directly incorporated into the 
    prompt context.
    """
    def __init__(
        self, 
        api_key: SecretStr = None,
        provider: str = "openai",
        model_name: str = MODEL_NAME,
        temperature: float = TEMPERATURE,
    ):
        self.provider = provider.lower() if provider.lower() in self._get_allowed_providers() else "openai" # Default provider (change in config.py)
        self.model_name = model_name # Default model name (change in config.py)
        self.temperature = temperature # Default temperature (change in config.py)
        self.api_key = api_key # api_key to access OpenAI API
        self.language = "English"
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """
        Initialize the NER model.
        """
        try:
            # Initializing the model with parameters
            if self.provider == "openai":
                self.model = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    api_key=self.api_key
                )
            elif self.provider == "huggingface":
                llm = HuggingFaceEndpoint(
                    repo_id=self.model_name,
                    task="text-generation",
                    temperature=self.temperature,
                    huggingfacehub_api_token=self.api_key
                )
                self.model = ChatHuggingFace(llm=llm)
            elif self.provider == "anthropic":
                self.model = ChatAnthropic(
                    model=self.model_name,
                    temperature=self.temperature,
                    api_key=self.api_key
                )
            elif self.provider == "ollama":
                self.model = ChatOllama(
                    model=self.model_name,
                    temperature=self.temperature
                )
            else:
                raise NERError(
                    f"Provider {self.provider} is not supported. Supported providers are: {', '.join(self._get_allowed_providers())}.",
                    error_type="initialization"
                )
        except Exception as e:
            raise NERError(
                f"Failed to initialize model with name {self.model_name}. {e}",
                error_type="initialization",
                original_error=e
            )
        
    def _get_allowed_providers(self) -> List[str]:
        return ["openai", "anthropic", "ollama", "huggingface"]

    def _determine_processing_mode(self, base_model: Type[BaseModel], tools_manager: ToolsManager) -> ProcessingMode:
        """
        Determine the processing mode based on the presence of tools and the base model.
        
        :param base_model: The base model class to use for structured output.
        :param tools_manager: The tools manager instance to use for agent-based processing.
        :return: The determined processing mode (unstructured, structured, or agentic)
        """
        if tools_manager and tools_manager.get_tools() and self.provider in ["openai", "anthropic"]:
            return ProcessingMode.AGENT_WITH_TOOLS
        elif base_model:
            return ProcessingMode.DIRECT_STRUCTURED
        else:
            return ProcessingMode.DIRECT_UNSTRUCTURED
        
    def _prepare_messages(
        self, 
        text: str, 
        tasks: List[str], 
        base_model: Type[BaseModel], 
        tools_manager: ToolsManager, 
        extra_info: str
    ) -> List[Any]:
        """ 
        Prepare messages for model/agent. 

        :param text: The input text to process.
        :param tasks: The list of tasks to perform.
        :param base_model: The base model class to use for structured output.
        :param tools_manager: The tools manager instance to use for agent-based processing.
        :param extra_info: Any extra information to include in the prompt.

        :return: The prepared messages for the model/agent.
        """
        try:
            tasks_formatted = "\n -".join(tasks)

            schema_str = ""
            parameter_desc = ""
            if base_model:
                schema_str = json.dumps(
                    base_model.model_json_schema(),
                    ensure_ascii=False,
                    indent=2
                )
                parameter_desc = "\n".join(
                    f" {name}: {field.description}"
                    for name, field in base_model.model_fields.items()
                )
            
            tools_descr = ""
            if tools_manager:
                tools = tools_manager.get_tools()
                tools_descr = "\n".join(f" - {tool.name}: {tool.description}" for tool in tools)

            sys_prompt = SYS_PROMPT.format(
                tasks=tasks_formatted,
                json_schema=schema_str,
                parameters=parameter_desc,
                tools=tools_descr
            )
            prompt = PROMPT.format(user_text=text, extra_info=extra_info)

            return [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=prompt)
            ]
        except Exception as e:
            raise NERError(
                f"Failed to prepare messages for the model. {e}",
                error_type="message_preparation",
                original_error=e
            )
        
    def _process_with_agent(self, messages: List[Any], tools_manager: ToolsManager) -> NERResult:
        """
        Process using agent with tools. 

        :param messages: The messages to process. Is a list containing system prompt and prompt for the LLM.
        :param tools_manager: The tools manager instance.
        :return: The result of the processing.
        """
        try:
            tools = tools_manager.get_tools()
            if not tools:
                return NERResult(
                    success=False,
                    error="No tools available for processing.",
                    mode=ProcessingMode.AGENT_WITH_TOOLS
                )
            logger.info(f"Initializing agent with {len(tools)} tools: {[t.name for t in tools]}")
            # TODO: Next step migration to LangGraph, initialize_agent will be deprecated
            agent = initialize_agent(
                tools,
                self.model,
                agent=AgentType.OPENAI_FUNCTIONS if self.provider == "openai" else AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True
            )

            result = agent.invoke(messages)

            agent_output = result.get('output', None)

            return NERResult(
                success=True,
                data=agent_output,
                mode=ProcessingMode.AGENT_WITH_TOOLS
            )
        except LangChainException as e:
            logger.error(f"LangChain error during agent processing: {e}")
            return NERResult(
                success=False,
                error=str(e),
                mode=ProcessingMode.AGENT_WITH_TOOLS
            )
        except Exception as e:
            logger.error(f"Error during agent processing: {e}")
            return NERResult(
                success=False,
                error=str(e),
                mode=ProcessingMode.AGENT_WITH_TOOLS
            )

    def _process_direct_structured(self, messages: List[Any], base_model: Type[BaseModel]) -> NERResult:
        """ 
        Process using direct model with structured output.

        :param messages: The messages to process.
        :param base_model: The base model class to use for structured output.
        :return: The result of the processing.
        """
        try:
            logger.info("Processing with direct structured output.")
            structured_llm = self.model.with_structured_output(base_model)
            result = structured_llm.invoke(messages)

            if result:
                return NERResult(
                    success=True,
                    data=result.model_dump_json(),
                    mode=ProcessingMode.DIRECT_STRUCTURED
                )
            else:
                return NERResult(
                    success=False,
                    error="No result returned from structured model.",
                    mode=ProcessingMode.DIRECT_STRUCTURED
                )
        except ValidationError as e:
            logger.error(f"Validation error during structured processing: {e}")
            return NERResult(
                success=False,
                error=f"Output validation failed: {e}",
                mode=ProcessingMode.DIRECT_STRUCTURED
            )
        except OpenAIError as e:
            logger.error(f"OpenAI error during structured processing: {e}")
            return NERResult(
                success=False,
                error=f"OpenAI error: {e}",
                mode=ProcessingMode.DIRECT_STRUCTURED
            )
        except Exception as e:
            logger.error(f"Error during structured processing: {e}")
            return NERResult(
                success=False,
                error=f"Unexpected error: {str(e)}",
                mode=ProcessingMode.DIRECT_STRUCTURED
            )
        
    def _process_direct_unstructured(self, messages: List[Any]) -> NERResult:
        """ 
        Process using direct model call without structured output.

        :param messages: The messages to process.
        :return: The result of the processing.
        """
        try:
            logger.info("Processing with direct unstructured output.")
            result = self.model.invoke(messages)

            if result and result.content:
                return NERResult(
                    success=True,
                    data=result.content,
                    mode=ProcessingMode.DIRECT_UNSTRUCTURED
                )
            else:
                return NERResult(
                    success=False,
                    error="No content returned from unstructured model.",
                    mode=ProcessingMode.DIRECT_UNSTRUCTURED
                )
        except OpenAIError as e:
            logger.error(f"OpenAI error during unstructured processing: {e}")
            return NERResult(
                success=False,
                error=f"OpenAI error: {e}",
                mode=ProcessingMode.DIRECT_UNSTRUCTURED
            )
        except Exception as e:
            logger.error(f"Error during unstructured processing: {e}")
            return NERResult(
                success=False,
                error=f"Unexpected error: {str(e)}",
                mode=ProcessingMode.DIRECT_UNSTRUCTURED
            )
        
    def _validate_structured_output(self, output: str, base_model: Type[BaseModel], mode: ProcessingMode, before_param_filling: bool = False) -> NERResult:
        """ 
        Validate and parse structured output from the model.

        :param output: The output string to validate.
        :param base_model: The base model class to use for validation.
        :param mode: The processing mode.
        :return: The result of the validation.
        """
        try:
            if isinstance(output, str):
                cleaned_output = output.strip()
                if cleaned_output.startswith('```json'):
                    cleaned_output = cleaned_output[7:]
                if cleaned_output.endswith('```'):
                    cleaned_output = cleaned_output[:-3]
                cleaned_output = cleaned_output.strip()
                try:
                    json_data = json.loads(cleaned_output)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    return NERResult(
                        success=False,
                        error=f"JSON parsing error: {e}",
                        mode=mode
                    )
                validated_result = base_model.model_validate(json_data)
            else:
                validated_result = base_model.model_validate(output)
            return NERResult(
                success=True,
                data=validated_result.model_dump_json(),
                mode=mode
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return NERResult(
                success=False,
                error=f"JSON parsing error: {e}",
                mode=mode
            )
        except ValidationError as e:
            if before_param_filling:
                logger.warning(f"Some parameters need filling: {e}")
            else:
                logger.error(f"Validation error during structured output processing: {e}")
            return NERResult(
                success=False,
                data=json_data if isinstance(output, str) else output,
                error=f"Validation error: {e}",
                mode=mode
            )
        except Exception as e:
            logger.error(f"Unexpected error during structured output processing: {e}")
            return NERResult(
                success=False,
                error=f"Unexpected error: {str(e)}",
                mode=mode
            )

    def _dynamic_type_casting(self, result: str, param: str, expected_type: Type, tools_manager: ToolsManager = None) -> Any:
        """
        Dynamically cast the result to the expected type using the LLM.

        :param result: The result string to cast.
        :param param: The parameter name requested.
        :param expected_type: The expected type to cast to.
        :return: The casted result.
        """
        
        tools_descr = ""
        tools = []
        if tools_manager and tools_manager.get_tools():
            tools = tools_manager.get_tools()
            tools_descr = "\n".join(f" - {tool.name}: {tool.description}" for tool in tools)

        prompt = PROMPT_TYPE_CASTING.format(
            input=result,
            param=param,
            expected_type=expected_type.__name__, 
            tools=tools_descr or "No tools available"
        )

        messages = [{"role": "user", "content": prompt}]
        if tools:
            agent = initialize_agent(
                tools,
                self.model,
                agent=AgentType.OPENAI_FUNCTIONS if self.provider == "openai" else AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True
            )
            response = agent.invoke(messages)
            output = response.get('output', None)
        else:
            response = self.model.invoke(messages)
            output = getattr(response, 'content', str(response))

        return output

    def _get_missing_params(self, result: NERResult, base_model: Type[BaseModel], tools_manager: ToolsManager = None) -> List[str]:
        """ 
        Get missing parameters from the NER result.

        :param result: The NER result to check for missing parameters.
        :param base_model: The base model class to use for validation.
        :param tools_manager: The tools manager to use for retrieving tools.
        :return: A list of missing parameter names.
        """
        optional_fields = []
        for name, field in base_model.model_fields.items():
            if not field.is_required():
                optional_fields.append(name)

        missing_params = []
        data_dict = json.loads(result.data) if isinstance(result.data, str) else result.data        

        if not data_dict:
            logger.warning("Result data is empty or None, skipping missing params check.")
            return missing_params

        for param in data_dict.keys():
            if data_dict[param] is None and param not in optional_fields:
                missing_params.append(param)
        
        logger.info(f"Missing parameters identified: {missing_params}")

        if not tools_manager or not tools_manager.get_tools():
            return missing_params
        tools_ = tools_manager.get_tools()
        tools_descr = "\n - ".join([f"Tool Name: {t.name}, Description: {t.description}" for t in tools_])
        prompt = PROMPT_PARAMS_RECOVERY.format(missing_params=missing_params, tools_descr=tools_descr)

        structured_llm = self.model.with_structured_output(NERMissingParameters)
        messages = [{"role": "user", "content": prompt}]
        response = structured_llm.invoke(messages)

        response = self._validate_structured_output(response, NERMissingParameters, ProcessingMode.DIRECT_STRUCTURED)
        response_data = json.loads(response.data) if isinstance(response.data, str) else response.data
        if response.success:
            missing_params = response_data.get("missing_params", [])
            logger.info(f"Missing parameters after tool recovery: {missing_params}")
        else:
            logger.error(f"Failed to recover missing parameters: {response.error}")
        response.data = json.dumps(response_data) if isinstance(response_data, dict) else response_data
        return missing_params
    
    def _ask_user_for_input(self, missing_params: List[str]) -> NERResult:
        """ 
        Ask user missing parameters.

        :param missing_params: List of the missing parameters.
        :return: The questions for the parameter requests.
        """

        questions = []

        for param in missing_params:
            questions.append({"parameter": {param}, "question": f"Please provide the value for '{param}'"})

        return NERResult(
            status="missing_parameter",
            questions=questions
        )
    
    def _ask_user_for_input_with_llm(self, result: NERResult, missing_params: List[str]) -> NERResult:
        """
        Ask user missing parameters using LLM.

        :param result: The result of the NER task.
        :param missing_params: List of the missing parameters.
        :return: The questions for the parameter requests.
        """

        questions = []

        for param in missing_params:
            prompt = PROMPT_PARAMETER_ASKING.format(current_schema=result.data, missing_parameter=param, language=self.language)
            messages = [{"role": "user", "content": prompt}]
            response = self.model.invoke(messages)
            question = getattr(response, 'content', str(response))
            questions.append({"parameter": param, "question": question})

        return NERResult(
            success=False,
            status="missing_parameter",
            questions=questions
        )

    def _post_processing_result(self, result: NERResult, tools_manager: ToolsManager, base_model: Type[BaseModel], ask_missing_params_with_llm: bool) -> NERResult:
        """ 
        Post-process the NER result to fill in any missing parameters. 

        :param result: The NER result to post-process.
        :param tools_manager: The tools manager to use for retrieving tools.
        :param base_model: The base model class to use for validation.
        :param ask_missing_params_with_llm: If True, the questions are generated by the LLM, otherwise use default questions.
        :return: The post-processed NER result.
        """
        # Identify missing parameters
        missing_params = self._get_missing_params(result=result, base_model=base_model, tools_manager=tools_manager) # Get missing parameters that are not possible to retrieve using the given tools.
        result_data = json.loads(result.data) if isinstance(result.data, str) else result.data
        
        if missing_params:
            params_req = self._ask_user_for_input_with_llm(result=result, missing_params=missing_params) if ask_missing_params_with_llm else self._ask_user_for_input(missing_params=missing_params)
            return params_req

        result.data = json.dumps(result_data)

        if self._get_missing_params(result=result, tools_manager=tools_manager, base_model=base_model):
            logger.warning("Some required parameters are still missing after post-processing.")
            result = self._post_processing_result(result=result, tools_manager=tools_manager, base_model=base_model, ask_missing_params_with_llm=ask_missing_params_with_llm)

        return result

    def chat(
        self, 
        text: str, 
        tasks: List[str], 
        base_model: Type[BaseModel] = None,
        tools_manager: ToolsManager = None,
        extra_info: str = "",
        ask_missing_params_with_llm: bool = False,
        user_inputs: dict = None,
        return_raw_result: bool = False
    ) -> str:
        """ 
        Process user text to extract structured information.

        :param text: Input text to process
        :param tasks: List of possible tasks
        :param base_model: Pydantic model class to use for structured output
        :param tools_manager: Manager for additional tools
        :param extra_info: Additional information to include inside the prompt
        :param ask_missing_params_with_llm: If True, the questions are generated by the LLM, otherwise use default questions.
        :param user_inputs: Dict of answers requested due to missing parameters or additional information.
        :param return_raw_result: If True, returns NERResult object instead of just data
        :return: Processed result as string/None or NERResult object
        """

        # Input checking
        try:
            if not text or not text.strip():
                # Handle empty or invalid input text
                error_result = NERResult(success=False, error="Input text is empty or invalid.")
                return error_result if return_raw_result else None
            
            if not tasks:
                # Handle missing tasks
                error_result = NERResult(success=False, error="No tasks provided for processing.")
                return error_result if return_raw_result else None

            # Detecting text language, in order to ask for missing parameters in the correct language
            self.language = pycountry.languages.get(alpha_2=detect(text)).name

            # Determining the processing mode:
            # 1. Unstructured processing (no BaseModel, no Tools)
            # 2. Structured processing (BaseModel present, no Tools), in this case the LLM will generate the JSON schema with consistent constraints
            # 3. Agentic processing (Tools present), in this case the LLM will use the tools to fill in the missing parameters (SUGGESTED MODE)
            mode = self._determine_processing_mode(base_model=base_model, tools_manager=tools_manager)
            logger.info(f"Determined processing mode: {mode.value}")

            # Preparing the messages for the LLM. With messages we mean the SYSTEM_MESSAGE and the HUMAN_MESSAGE that will be sent to the model.
            # The tool manager is sent to provide the LLM the necessary tools for processing.
            # The base model is sent to provide the requested JSON schema.
            # Extra info are used to enrich the prompt with context (USE ONLY IF NECESSARY).
            messages = self._prepare_messages(text=text, tasks=tasks, base_model=base_model, tools_manager=tools_manager, extra_info=extra_info)

            if mode == ProcessingMode.AGENT_WITH_TOOLS:
                result = self._process_with_agent(messages, tools_manager) # Using agent with tools
            elif mode == ProcessingMode.DIRECT_STRUCTURED:
                result = self._process_direct_structured(messages, base_model) # Direct structured processing
            else:
                result = self._process_direct_unstructured(messages) # Direct unstructured processing

            if user_inputs:
                result_data = json.loads(result.data) if isinstance(result.data, str) else result.data
                for param, answer in user_inputs.items():
                    user_input = self._dynamic_type_casting(result=answer, param=param, tools_manager=tools_manager, expected_type=base_model.model_fields[param].annotation)
                    result_data[param] = user_input
                result.data = json.dumps(result_data)

            result = self._validate_structured_output(output=json.loads(result.data) if isinstance(result.data, str) else result.data, base_model=base_model, mode=mode, before_param_filling=True) if base_model else result

            # Check missing parameters
            result = self._post_processing_result(result=result, tools_manager=tools_manager, base_model=base_model, ask_missing_params_with_llm=ask_missing_params_with_llm)

            if result.status == "missing_parameter":
                return result if return_raw_result else result.to_dict()

            # Validating structured output if base_model is present
            result = self._validate_structured_output(output=json.loads(result.data) if isinstance(result.data, str) else result.data, base_model=base_model, mode=mode) if base_model else result

            if return_raw_result:
                return result # Return a NERResult object
            else:
                return json.loads(result.data) if result.success else None # Return structured data if available
        except NERError as e:
            logger.error(f"NER processing error: {e}")
            return NERResult(success=False, error=str(e), mode=mode) if return_raw_result else None
        except Exception as e:
            logger.error(f"Unexpected error during NER processing: {e}")
            error_result = NERResult(success=False, error=str(e), mode=mode)
            return error_result if return_raw_result else None

    def detect_multiactivities(self, text: str, prompt: str = "", examples: str = "No other examples available") -> List[str]:
        """
        Detect multiple activities in the text.

        :param text: The input text to analyze.
        :param prompt: The prompt to use for the LLM. Use specific prompt for multi-activity detection only for particular cases, otherwise use the default prompt.
        :param examples: Additional examples to provide context for the LLM. Not needed if prompt is provided.
        :return: A list of detected activities.
        """
        if prompt == "":
            prompt = PROMPT_MULTIACTIVITY.format(input=text, examples=examples)

        messages = [{"role": "user", "content": prompt}]
        llm = self.model.with_structured_output(MultiActivity)
        
        response = llm.invoke(messages)
        return response.activities

    def base_model_selector(self, base_models: List[Type[BaseModel]], text: str) -> Type[BaseModel]:
        """
        Given a set of BaseModel, select the most appropriate one based on the input text.

        :param base_models: A list of BaseModel classes to choose from.
        :param text: The input text to analyze.
        :return: The most appropriate BaseModel class.
        """
        models = "\n".join(f" - {i}: {base_model.model_json_schema()}" for i, base_model in enumerate(base_models))

        prompt = PROMPT_BASE_MODEL_SELECTOR.format(base_models=models, text=text)
        messages = [{"role": "user", "content": prompt}]

        model = self.model.with_structured_output(BaseModelSelector)

        response = model.invoke(messages)
        return base_models[response.idx]

    def set_model_params(self, provider: str = PROVIDER, model_name: str = MODEL_NAME, temperature: float = TEMPERATURE) -> None:
        """
        Set the model parameters for the NER instance.

        :param model_name: The name of the model to use.
        :param temperature: The temperature to use for sampling.
        """
        try:
            self.provider = provider.lower() if provider.lower() in self._get_allowed_providers() else "openai"
            self.model_name = model_name
            self.temperature = temperature
            self._initialize_model()
            logger.info(f"Model parameters set: provider={self.provider}, model_name={self.model_name}, temperature={self.temperature}")
        except Exception as e:
            raise NERError(
                f"Failed to set model parameters: {e}",
                error_type="parameter_setting",
                original_error=e
            )
        
    def get_model_params(self) -> dict:
        """
        Get the current model parameters.

        :return: A dictionary containing the model name and temperature.
        """
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "api_key_set": bool(self.api_key)
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Performs a health check of the NER system.

        :return: A dictionary containing the health check results.
        """
        try:
            test_messages = [HumanMessage(content="Test message for health check.")]
            result = self.model.invoke(test_messages)

            return {
                "status": "healthy",
                "model_name": self.model_name,
                "temperature": self.temperature
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_name": self.model_name,
                "temperature": self.temperature
            }
            
    def __str__(self) -> str:
        return f"NER(provider={self.provider}, model_name={self.model_name}, temperature={self.temperature})"