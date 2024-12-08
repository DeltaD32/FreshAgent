class Agent(BaseModel):
    config: AgentConfig
    llm: Optional[LLMInterface] = None
    message_queue: List[Message] = []
    
    class Config:
        arbitrary_types_allowed = True

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Custom dictionary serialization."""
        try:
            return {
                "config": self.config.dict(),
                "message_queue": [msg.dict() for msg in self.message_queue]
            }
        except Exception as e:
            logger.error(f"Error serializing agent {self.config.name}: {str(e)}")
            raise ValueError(f"Failed to serialize agent: {str(e)}")

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def role(self) -> str:
        return self.config.role 

class AgentConfig(BaseModel):
    name: str
    role: str
    description: str = ""
    is_orchestrator: bool = False
    model: str = "mistral"
    provider: str = "ollama"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    system_prompt: str = ""
    
    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Custom dictionary serialization."""
        try:
            return {
                "name": self.name,
                "role": self.role,
                "description": self.description,
                "is_orchestrator": self.is_orchestrator,
                "model": self.model,
                "provider": self.provider,
                "api_key": self.api_key,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "system_prompt": self.system_prompt
            }
        except Exception as e:
            logger.error(f"Error serializing agent config {self.name}: {str(e)}")
            raise ValueError(f"Failed to serialize agent config: {str(e)}") 