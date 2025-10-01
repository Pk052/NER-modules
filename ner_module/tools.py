import logging
from langchain_core.tools import BaseTool, tool
from typing import List, Callable
from functools import wraps
from .utils import ToolError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolsManager:
    """
    This class is designed to manage and organize tools for the application.
    With tools we mean functions or methods that can be used to perform specific tasks.
    These tools can be called by the LLM agent to perform actions based on user input.

    With tools the LLM agent can extend its capabilities and interact with the entire system more effectively.

    For example, a tool could include a function to query a database, call an external API, or perform complex calculations.
    In this way, the LLM call can leverage these tools to perform actions that require external knowledge or capabilities.

    **IMPORTANT**: Each function or tool function that is added as a tool must be commented with its purpose and usage. This is **ESSENTIAL**
    for the correct calling of the tool by the agent. In case the function is not properly documented, the agent will throw an error.
    """
        
    def __init__(self, host: str = None, bearer_token: str = None):
        self.host = host # The host URL for the tools manager, if the tools need to make API calls
        self.token = bearer_token # The bearer token for authentication, if required
        self.tools: List[BaseTool] = [] # The list of tools

    @staticmethod
    def _logging_wrapper(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                logger.info(f"Calling tool function: {fn.__name__} with args: {args}, kwargs: {kwargs}")
                result = fn(*args, **kwargs)
                logger.info(f"Tool function: {fn.__name__} returned: {result}")
            except Exception as e:
                logger.error(f"Error in tool function: {fn.__name__} with args: {args}, kwargs: {kwargs}. Error: {e}")
                raise ToolError(f"Error in tool function: {fn.__name__}", tool_name=fn.__name__, original_error=e)
            return result
        return wrapper

    def add_tools(self, tool: BaseTool) -> None:
        """
        Adds a tool to the tools manager.

        :param tool: The tool to add. Is a function or method that performs a specific task, that has been wrapped as a BaseTool using the `tool` decorator.
        """
        self.tools.append(tool)

    def remove_tools(self, tool: BaseTool) -> None:
        """
        Removes a tool from the tools manager.

        :param tool: The tool to remove.
        """
        self.tools.remove(tool)

    def clear_tools(self) -> None:
        """
        Clears all tools from the tools manager.
        """
        self.tools.clear()

    def get_tools(self) -> List[BaseTool]:
        """
        Returns a list of all tools in the tools manager.
        """
        return self.tools.copy()

    def get_tool_by_name(self, name: str) -> BaseTool:
        """
        Returns a tool from the tools manager by its name.

        :param name: The name of the tool to retrieve.
        :return: The tool with the specified name, or None if not found.
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def fn_to_tool_fn(self, fn: callable) -> BaseTool:
        """
        Converts a function to a tool function.

        :params fn: The function to convert. In this way, the function will be wrapped as a BaseTool using the `tool` decorator.
        """
        logged_fn = self._logging_wrapper(fn) # Wrapping function with logging, in this way we can track which functions are being called by the LLM agent
        return tool(logged_fn)

    def add_fn_to_tools(self, fn: callable) -> None:
        """
        Adds a function as a tool to the tools manager.
        
        :params fn: The function to add. Performs the same logic of the `fn_to_tool_fn` method, but the function is added directly in the tools list.
        """
        self.add_tools(self.fn_to_tool_fn(fn))

    def __len__(self) -> int:
        return len(self.tools)

    def __str__(self) -> str:
        return f"ToolsManager with {len(self.tools)} tools: {[tool.name for tool in self.tools]}"
