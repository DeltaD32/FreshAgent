from typing import Dict, Optional, List, Tuple, Any
from pydantic import BaseModel
from llm_interface import LLMInterface
from datetime import datetime
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

class Message(BaseModel):
    sender: str
    receiver: str
    content: str
    message_type: str = "task"  # task, response, or status
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = {}

class AgentConfig(BaseModel):
    name: str
    description: str
    capabilities: List[str]
    system_prompt: str
    status: str = "initialized"
    health_check_timestamp: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    is_orchestrator: bool = False

    def perform_health_check(self) -> Tuple[bool, str]:
        """Perform health check on the agent configuration."""
        try:
            if not self.name or not isinstance(self.name, str):
                return False, "Invalid agent name"
            if not self.description or not isinstance(self.description, str):
                return False, "Invalid description"
            if not isinstance(self.capabilities, list) or not self.capabilities:
                return False, "Invalid capabilities"
            if not self.system_prompt or not isinstance(self.system_prompt, str):
                return False, "Invalid system prompt"
            
            self.health_check_timestamp = datetime.now()
            self.status = "healthy"
            return True, "Health check passed"
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            self.status = "error"
            return False, f"Health check failed: {str(e)}"

class Agent(BaseModel):
    config: AgentConfig
    llm: LLMInterface
    message_queue: List[Message] = []
    
    class Config:
        arbitrary_types_allowed = True

    def dict(self, *args, **kwargs):
        """Custom dictionary serialization."""
        return {
            "config": self.config.dict(),
            "message_queue": [msg.dict() for msg in self.message_queue]
        }

    async def process_message(self, message: Message) -> Optional[Message]:
        """Process an incoming message and generate a response."""
        if message.receiver != self.config.name:
            return None
            
        # Construct prompt with message context
        prompt = f"""Process this message as {self.config.name} with capabilities: {', '.join(self.config.capabilities)}
        
        Message from {message.sender}: {message.content}
        
        Respond appropriately based on your capabilities and role."""
        
        try:
            response_content = await self.llm.generate(
                prompt=prompt,
                system_prompt=self.config.system_prompt
            )
            
            return Message(
                sender=self.config.name,
                receiver=message.sender,
                content=response_content,
                message_type="response",
                metadata={"original_message_id": id(message)}
            )
        except Exception as e:
            logger.error(f"Error processing message in agent {self.config.name}: {str(e)}")
            return Message(
                sender=self.config.name,
                receiver=message.sender,
                content=f"Error processing message: {str(e)}",
                message_type="error",
                metadata={"error": str(e)}
            )

    async def execute_task(self, task: str) -> str:
        """Execute a task using the agent's capabilities."""
        try:
            return await self.llm.generate(
                prompt=task,
                system_prompt=self.config.system_prompt
            )
        except Exception as e:
            logger.error(f"Error executing task in agent {self.config.name}: {str(e)}")
            return f"Error executing task: {str(e)}"

class OrchestratorAgent(Agent):
    """Specialized agent for task delegation and coordination."""
    
    def dict(self, *args, **kwargs):
        """Custom dictionary serialization for orchestrator."""
        base_dict = super().dict(*args, **kwargs)
        base_dict["is_orchestrator"] = True
        return base_dict
    
    async def delegate_task(self, task: str, available_agents: Dict[str, Agent]) -> str:
        """Analyze task and delegate to appropriate agents."""
        try:
            # First, analyze the task to determine required capabilities
            analysis_prompt = f"""Analyze this task and determine which specialized agents should handle different parts:
            
            Task: {task}
            
            Available agents and their capabilities:
            {self._format_agent_capabilities(available_agents)}
            
            Respond in JSON format:
            {{
                "subtasks": [
                    {{
                        "agent": "agent_name",
                        "task": "specific task description",
                        "required_capability": "main_capability_needed"
                    }}
                ]
            }}"""
            
            analysis_response = await self.llm.generate(analysis_prompt)
            
            try:
                # Parse the delegation plan
                plan = json.loads(analysis_response)
                
                # Execute all subtasks concurrently
                subtask_messages = []
                for subtask in plan["subtasks"]:
                    agent_name = subtask["agent"]
                    if agent_name in available_agents:
                        message = Message(
                            sender=self.config.name,
                            receiver=agent_name,
                            content=subtask["task"],
                            message_type="task",
                            metadata={"required_capability": subtask["required_capability"]}
                        )
                        subtask_messages.append(message)
                
                # Send all messages and gather responses
                responses = await asyncio.gather(
                    *[available_agents[msg.receiver].process_message(msg) 
                      for msg in subtask_messages]
                )
                
                # Synthesize final response
                synthesis_prompt = f"""Synthesize these agent responses into a coherent final response:
                
                Original task: {task}
                
                Agent responses:
                {self._format_agent_responses(responses)}
                
                Provide a unified response that combines all agent contributions."""
                
                final_response = await self.llm.generate(synthesis_prompt)
                return final_response
                
            except json.JSONDecodeError:
                logger.error(f"Error parsing task delegation plan in orchestrator {self.config.name}")
                return f"Error: Could not parse task delegation plan. Falling back to direct execution: {await self.execute_task(task)}"
            except Exception as e:
                logger.error(f"Error in task delegation for orchestrator {self.config.name}: {str(e)}")
                return f"Error in task delegation: {str(e)}"
        except Exception as e:
            logger.error(f"Error in delegate_task for orchestrator {self.config.name}: {str(e)}")
            return f"Error in task delegation: {str(e)}"
    
    def _format_agent_capabilities(self, agents: Dict[str, Agent]) -> str:
        """Format agent capabilities for prompt."""
        return "\n".join(
            f"- {agent.config.name}: {', '.join(agent.config.capabilities)}"
            for agent in agents.values()
        )
    
    def _format_agent_responses(self, responses: List[Message]) -> str:
        """Format agent responses for synthesis prompt."""
        return "\n".join(
            f"{response.sender}: {response.content}"
            for response in responses if response is not None
        )

class AgentFactory:
    def __init__(self, default_model: str = "mistral"):
        self.default_model = default_model
        try:
            self.llm_interface = LLMInterface(model_name=default_model)
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM interface: {str(e)}")
        self.agents: Dict[str, Agent] = {}
        self._last_health_check: datetime = datetime.now()
        self._health_status: str = "initialized"

    def create_agent(self, config: AgentConfig) -> Agent:
        """Create a new agent with specified configuration and perform health check."""
        try:
            # Validate configuration
            is_healthy, message = config.perform_health_check()
            if not is_healthy:
                raise ValueError(f"Invalid agent configuration: {message}")
            
            # Create appropriate agent class
            agent_class = OrchestratorAgent if config.is_orchestrator else Agent
            
            # Initialize agent
            agent = agent_class(
                config=config,
                llm=self.llm_interface
            )
            
            # Verify agent initialization
            if not agent or not agent.config or not agent.llm:
                raise ValueError("Agent initialization failed")
            
            # Store agent
            self.agents[config.name] = agent
            return agent
            
        except Exception as e:
            raise ValueError(f"Agent creation failed: {str(e)}")

    async def perform_health_check(self) -> Tuple[bool, Dict[str, str]]:
        """Perform health check on all agents."""
        self._last_health_check = datetime.now()
        results = {}
        all_healthy = True
        
        for name, agent in self.agents.items():
            is_healthy, message = agent.config.perform_health_check()
            results[name] = message
            if not is_healthy:
                all_healthy = False
        
        self._health_status = "healthy" if all_healthy else "degraded"
        return all_healthy, results

    async def repair_agent(self, agent_name: str) -> Tuple[bool, str]:
        """Attempt to repair an unhealthy agent."""
        agent = self.agents.get(agent_name)
        if not agent:
            return False, f"Agent {agent_name} not found"
        
        try:
            # Generate new configuration while keeping the name
            repair_prompt = f"""Fix the following agent configuration:
            Current name: {agent.config.name}
            Current role: {agent.config.description}
            Current capabilities: {', '.join(agent.config.capabilities)}
            Current issues: {agent.config.last_error}
            
            Provide a fixed configuration that maintains the agent's core purpose while resolving any issues."""
            
            response = await self.llm_interface.generate(repair_prompt)
            
            # Parse the response and update the agent
            new_config = await self._parse_agent_config(response.content, agent.config.name)
            is_healthy, message = new_config.perform_health_check()
            
            if is_healthy:
                self.agents[agent_name] = Agent(config=new_config, llm=self.llm_interface)
                return True, "Agent repaired successfully"
            else:
                return False, f"Repair failed: {message}"
                
        except Exception as e:
            return False, f"Repair failed: {str(e)}"

    async def _parse_agent_config(self, config_text: str, existing_name: Optional[str] = None) -> AgentConfig:
        """Parse agent configuration from text with error handling."""
        try:
            config_dict = {}
            current_key = None
            current_value = []
            
            for line in config_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if any(line.startswith(k + ":") for k in ["Name", "Description", "Capabilities", "System Prompt"]):
                    if current_key and current_value:
                        config_dict[current_key] = ' '.join(current_value)
                    
                    key, value = line.split(":", 1)
                    current_key = key.strip().lower().replace(" ", "_")
                    current_value = [value.strip()]
                else:
                    if current_key:
                        current_value.append(line)
            
            if current_key and current_value:
                config_dict[current_key] = ' '.join(current_value)
            
            return AgentConfig(
                name=existing_name or config_dict.get('name', f"Agent_{len(self.agents)}"),
                description=config_dict.get('description', ""),
                capabilities=config_dict.get('capabilities', "").split(','),
                system_prompt=config_dict.get('system_prompt', "You are a helpful AI assistant.")
            )
            
        except Exception as e:
            raise ValueError(f"Failed to parse agent configuration: {str(e)}")

    def get_agent(self, name: str) -> Optional[Agent]:
        """Retrieve an existing agent by name."""
        return self.agents.get(name)

    async def create_specialized_agent(self, task_description: str) -> Agent:
        """Create a specialized agent based on task description."""
        try:
            # First, analyze the task to determine required capabilities
            analysis_prompt = f"""Analyze this task and determine the required agent capabilities:
            Task: {task_description}
            
            Provide a detailed analysis of:
            1. Required capabilities
            2. Specialized knowledge areas
            3. Key responsibilities
            4. Interaction patterns
            
            Then create a specialized agent configuration in this exact format:
            Name: [descriptive name based on primary capability]
            Description: [detailed description of agent's role and responsibilities]
            Capabilities: [comma-separated list of specific capabilities]
            System Prompt: [detailed system prompt that guides agent behavior]"""

            response = await self.llm_interface.generate(analysis_prompt)
            
            # Enhanced parsing with validation
            config = await self._parse_agent_config_enhanced(response, task_description)
            
            # Validate configuration before creating agent
            is_healthy, message = config.perform_health_check()
            if not is_healthy:
                logger.warning(f"Initial configuration unhealthy: {message}. Attempting repair...")
                config = await self._repair_config(config, task_description)
            
            # Create and verify agent
            agent = self.create_agent(config)
            
            # Verify agent capabilities
            verification_prompt = f"""Verify if this agent configuration is suitable for the task:
            Task: {task_description}
            Agent Configuration: {json.dumps(config.dict(), indent=2)}
            
            Respond with:
            1. Capability match score (0-100)
            2. Missing capabilities (if any)
            3. Suggested improvements"""
            
            verification = await self.llm_interface.generate(verification_prompt)
            
            # Log verification results
            logger.info(f"Agent verification for {config.name}: {verification}")
            
            return agent
            
        except Exception as e:
            logger.error(f"Error creating specialized agent: {str(e)}")
            # Create fallback agent with enhanced error handling
            return await self._create_fallback_agent(task_description, str(e))

    async def _parse_agent_config_enhanced(self, response: str, task_description: str) -> AgentConfig:
        """Enhanced parsing of agent configuration with validation and cleanup."""
        try:
            config_dict = {}
            sections = ["name", "description", "capabilities", "system_prompt"]
            current_section = None
            current_content = []
            
            # Split response into lines and clean
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            for line in lines:
                # Check for section headers
                lower_line = line.lower()
                if any(lower_line.startswith(f"{section}:") for section in sections):
                    # Save previous section
                    if current_section and current_content:
                        config_dict[current_section] = ' '.join(current_content).strip()
                    
                    # Start new section
                    section_split = line.split(':', 1)
                    current_section = section_split[0].lower().replace(' ', '_')
                    current_content = [section_split[1].strip()] if len(section_split) > 1 else []
                elif current_section:
                    current_content.append(line)
            
            # Save final section
            if current_section and current_content:
                config_dict[current_section] = ' '.join(current_content).strip()
            
            # Validate and clean capabilities
            capabilities = config_dict.get('capabilities', '')
            capabilities = [cap.strip() for cap in capabilities.split(',') if cap.strip()]
            
            # Ensure minimum required fields
            name = config_dict.get('name', '').strip() or f"Agent_{len(self.agents)}"
            description = config_dict.get('description', '').strip() or f"Specialized agent for {task_description[:100]}"
            system_prompt = config_dict.get('system_prompt', '').strip() or f"You are a specialized AI assistant for: {task_description}"
            
            return AgentConfig(
                name=name,
                description=description,
                capabilities=capabilities or ["general_task_execution"],
                system_prompt=system_prompt,
                status="initialized"
            )
            
        except Exception as e:
            logger.error(f"Error parsing agent configuration: {str(e)}")
            raise ValueError(f"Failed to parse agent configuration: {str(e)}")

    async def _repair_config(self, config: AgentConfig, task_description: str) -> AgentConfig:
        """Attempt to repair an invalid agent configuration."""
        try:
            repair_prompt = f"""Fix the following agent configuration issues:
            Current Configuration:
            {json.dumps(config.dict(), indent=2)}
            
            Task Description: {task_description}
            
            Create an improved configuration that:
            1. Maintains the agent's core purpose
            2. Fixes any validation issues
            3. Enhances capabilities for the task
            4. Provides a more robust system prompt
            
            Provide the fixed configuration in the standard format:
            Name: [name]
            Description: [description]
            Capabilities: [capabilities]
            System Prompt: [system prompt]"""
            
            response = await self.llm_interface.generate(repair_prompt)
            new_config = await self._parse_agent_config_enhanced(response, task_description)
            
            # Verify the repaired config
            is_healthy, message = new_config.perform_health_check()
            if not is_healthy:
                logger.error(f"Config repair failed: {message}")
                raise ValueError(f"Could not repair configuration: {message}")
            
            return new_config
            
        except Exception as e:
            logger.error(f"Error repairing configuration: {str(e)}")
            raise ValueError(f"Configuration repair failed: {str(e)}")

    async def _create_fallback_agent(self, task_description: str, error_context: str) -> Agent:
        """Create a fallback agent when specialized creation fails."""
        try:
            # Create a simplified but robust configuration
            config = AgentConfig(
                name=f"FallbackAgent_{len(self.agents)}",
                description=f"Fallback agent for: {task_description[:100]}",
                capabilities=["general_task_execution", "error_recovery", "task_adaptation"],
                system_prompt=f"""You are a resilient AI assistant created as a fallback for the task:
                {task_description}
                
                Previous error context: {error_context}
                
                Focus on:
                1. Providing basic task execution capabilities
                2. Adapting to task requirements dynamically
                3. Maintaining stable operation
                4. Reporting any issues encountered
                
                Approach tasks cautiously and report any difficulties encountered.""",
                status="fallback"
            )
            
            return self.create_agent(config)
            
        except Exception as e:
            logger.error(f"Critical error creating fallback agent: {str(e)}")
            raise RuntimeError(f"Could not create fallback agent: {str(e)}")

    async def regenerate_agent(self, agent_name: str) -> Tuple[bool, str]:
        """Regenerate an existing agent with improved capabilities."""
        try:
            agent = self.agents.get(agent_name)
            if not agent:
                raise ValueError(f"Agent {agent_name} not found")
            
            # Analyze current configuration and performance
            analysis_prompt = f"""Analyze and improve this agent configuration:
            Current Configuration:
            {json.dumps(agent.config.dict(), indent=2)}
            
            Provide an enhanced configuration that:
            1. Maintains successful aspects
            2. Improves weak areas
            3. Adds new relevant capabilities
            4. Enhances the system prompt
            
            Respond in the standard configuration format."""
            
            response = await self.llm_interface.generate(analysis_prompt)
            new_config = await self._parse_agent_config_enhanced(response, agent.config.description)
            
            # Validate new configuration
            is_healthy, message = new_config.perform_health_check()
            if not is_healthy:
                new_config = await self._repair_config(new_config, agent.config.description)
            
            # Create new agent with preserved name
            new_config.name = agent_name
            new_agent = self.create_agent(new_config)
            
            # Replace old agent
            self.agents[agent_name] = new_agent
            
            return True, "Agent regenerated successfully"
            
        except Exception as e:
            logger.error(f"Error regenerating agent {agent_name}: {str(e)}")
            return False, f"Agent regeneration failed: {str(e)}" 