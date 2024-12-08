from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = {}

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Custom dictionary serialization."""
        try:
            return {
                "role": self.role,
                "content": self.content,
                "timestamp": self.timestamp.isoformat(),
                "metadata": self.metadata
            }
        except Exception as e:
            logger.error(f"Error serializing message: {str(e)}")
            raise ValueError(f"Failed to serialize message: {str(e)}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        try:
            if isinstance(data.get("timestamp"), str):
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            return cls(**data)
        except Exception as e:
            logger.error(f"Error creating message from dictionary: {str(e)}")
            raise ValueError(f"Failed to create message from dictionary: {str(e)}") 