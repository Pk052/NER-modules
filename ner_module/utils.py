from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class NERMissingParameters(BaseModel):
    missing_params: List[str]

class BaseModelSelector(BaseModel):
    idx: int
    
class MultiActivity(BaseModel):
    activities: list[str] = Field(..., description="Array of activity strings.")

class ProcessingMode(Enum):
    """ Enum to define different processing modes for the NER system """
    DIRECT_STRUCTURED = "direct_structured" # Direct model call with structured output
    AGENT_WITH_TOOLS = "agent_with_tools" # Agent-based processing with tools
    DIRECT_UNSTRUCTURED = "direct_unstructured" # Direct model call with unstructured output

class NERResult:
    def __init__(self, success: bool, status: str = "ok", questions: List[str] = None, data: Any = None, error: str = None, mode: ProcessingMode = None):
        self.success = success
        self.status = status
        self.questions = questions
        self.data = data
        self.error = error
        self.mode = mode

    def to_dict(self) -> Dict[str, Any]:
        if self.questions:
            return {
                "success": self.success,
                "status": self.status,
                "questions": self.questions,
                "data": self.data,
                "error": self.error,
                "mode": self.mode
            }
        else:
            return {
                "success": self.success,
                "status": self.status,
                "data": self.data,
                "error": self.error,
                "mode": self.mode
            }
    
class NERError(Exception):
    """ Custom exception for NER errors """
    def __init__(self, message: str, error_type: str = "general", original_error: Exception = None):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error

class ToolError(Exception):
    def __init__(self, message: str, tool_name: str = None, original_error: Exception = None):
        super().__init__(message)
        self.tool_name = tool_name
        self.original_error = original_error