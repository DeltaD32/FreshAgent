import streamlit as st
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import shutil
from pydantic import BaseModel
import asyncio
from llm_interface import LLMInterface
from agent_factory import Agent, AgentConfig, AgentFactory, OrchestratorAgent, Message
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_llm_interface(model_name: str = "mistral", provider: str = "ollama", api_key: Optional[str] = None) -> LLMInterface:
    """Initialize LLM interface synchronously."""
    # Use cached LLM interface if available
    if not hasattr(st.session_state, 'llm_interface'):
        try:
            llm = LLMInterface(model_name=model_name, provider=provider, api_key=api_key)
            st.session_state.llm_interface = llm
            logger.info("Created new LLM interface and cached it")
        except Exception as e:
            logger.error(f"Error initializing LLM interface: {str(e)}")
            raise
    return st.session_state.llm_interface

def generate_description_sync(name: str, base_description: str = "") -> str:
    """Generate an enhanced project description using AI (synchronous version)."""
    try:
        llm = init_llm_interface()
        prompt = f"""Given the project name '{name}' and base description '{base_description}', 
        generate a comprehensive technical project description. Include:
        1. Project overview
        2. Key technical features
        3. Potential use cases
        4. Technical requirements
        Keep the description professional and focused on technical aspects."""
        
        response = llm.generate_sync(prompt)
        return response.strip()
    except Exception as e:
        logger.error(f"Error generating description: {str(e)}")
        return base_description

class Project(BaseModel):
    name: str
    description: str
    created_at: datetime = datetime.now()
    agents: Dict[str, Agent] = {}
    conversation_history: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {
        "model_provider": "ollama",
        "current_model": "mistral",
        "openai_api_key": None,
        "openai_model": None
    }
    feature_bible: Dict[str, Any] = {
        "features": [],
        "last_updated": None,
        "pending_changes": [],
        "version": "1.0"
    }
    llm_interface: Optional[LLMInterface] = None
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.llm_interface = init_llm_interface(
            model_name=self.metadata["current_model"],
            provider=self.metadata["model_provider"],
            api_key=self.metadata.get("openai_api_key")
        )
        logger.info(f"Project {self.name} initialized with LLM interface")

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Custom dictionary serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "agents": {
                name: agent.dict() if hasattr(agent, 'dict') else str(agent)
                for name, agent in self.agents.items()
            },
            "conversation_history": self.conversation_history,
            "metadata": self.metadata,
            "feature_bible": self.feature_bible
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        """Create project from dictionary."""
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)

class ProjectManager:
    """Manages project creation, loading, and storage."""
    
    def __init__(self):
        """Initialize project manager."""
        self.projects_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects")
        os.makedirs(self.projects_dir, exist_ok=True)
        logger.info(f"Initialized ProjectManager with projects directory: {self.projects_dir}")

    def save_project(self, project: Project):
        """Save project to disk."""
        try:
            logger.info(f"Saving project {project.name}")
            project_dir = os.path.join(self.projects_dir, project.name)
            os.makedirs(project_dir, exist_ok=True)
            
            project_file = os.path.join(project_dir, "project.json")
            project_data = project.dict()
            
            with open(project_file, "w") as f:
                json.dump(project_data, f, indent=2)
                
            logger.info(f"Project {project.name} saved successfully")
        except Exception as e:
            logger.error(f"Error saving project: {str(e)}")
            raise ValueError(f"Failed to save project: {str(e)}")

    def load_project(self, name: str) -> Optional[Project]:
        """Load project from disk."""
        try:
            logger.info(f"Loading project {name}")
            project_dir = os.path.join(self.projects_dir, name)
            project_file = os.path.join(project_dir, "project.json")
            
            if not os.path.exists(project_file):
                logger.warning(f"Project file not found: {project_file}")
                return None
                
            with open(project_file, "r") as f:
                project_data = json.load(f)
                
            project = Project.from_dict(project_data)
            logger.info(f"Project {name} loaded successfully")
            return project
            
        except Exception as e:
            logger.error(f"Error loading project {name}: {str(e)}")
            return None

    def list_projects(self) -> List[str]:
        """List all available projects."""
        try:
            if not os.path.exists(self.projects_dir):
                return []
            return [d for d in os.listdir(self.projects_dir) 
                   if os.path.isdir(os.path.join(self.projects_dir, d))]
        except Exception as e:
            logger.error(f"Error listing projects: {str(e)}")
            return []

    def delete_project(self, name: str):
        """Delete a project and its files."""
        try:
            project_dir = os.path.join(self.projects_dir, name)
            if os.path.exists(project_dir):
                shutil.rmtree(project_dir)
                logger.info(f"Project {name} deleted successfully")
            else:
                logger.warning(f"Project directory not found: {project_dir}")
        except Exception as e:
            logger.error(f"Error deleting project: {str(e)}")
            raise ValueError(f"Error deleting project: {str(e)}")

def main():
    st.title("FreshAgent - Local LLM Agent System")
    
    # Initialize session state
    if 'project_manager' not in st.session_state:
        st.session_state.project_manager = ProjectManager()
        logger.info("Initialized ProjectManager in session state")
    
    if 'form_state' not in st.session_state:
        st.session_state.form_state = {
            'name': '',
            'description': '',
            'enhanced_description': ''
        }
    
    # Sidebar for project management
    with st.sidebar:
        st.header("Project Management")
        
        # Project creation wizard
        with st.expander("Create New Project", expanded=False):
            # Basic Info
            st.subheader("Basic Information")
            project_name = st.text_input("Project Name", key="new_project_name")
            project_description = st.text_area("Project Description", key="new_project_description")
            
            # Project Type and Keywords
            st.subheader("Project Configuration")
            project_type = st.selectbox(
                "Project Type",
                ["AI Agent System", "Web Application", "API Service", 
                 "Data Processing", "Machine Learning", "Other"],
                key="new_project_type"
            )
            
            keywords = st.multiselect(
                "Keywords",
                ["AI", "Machine Learning", "Web", "API",
                 "Data Processing", "Automation", "Integration",
                 "Security", "Analytics", "Real-time"],
                key="new_project_keywords"
            )
            
            # Model Configuration
            st.subheader("Model Configuration")
            model_provider = st.selectbox(
                "Model Provider",
                ["ollama", "openai"],
                key="new_project_model_provider"
            )
            
            if model_provider == "ollama":
                model = st.selectbox(
                    "Model",
                    ["mistral", "llama2", "codellama"],
                    key="new_project_model_ollama"
                )
                api_key = None
            else:
                model = st.selectbox(
                    "Model",
                    ["gpt-3.5-turbo", "gpt-4"],
                    key="new_project_model_openai"
                )
                api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    key="new_project_api_key"
                )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Create Project", key="create_project_btn"):
                    try:
                        # Use the enhanced description if available
                        final_description = (st.session_state.form_state['enhanced_description'] 
                                          if st.session_state.form_state['enhanced_description'] 
                                          else project_description)
                        
                        # Create project with full configuration
                        project = Project(
                            name=project_name,
                            description=final_description,
                            metadata={
                                "model_provider": model_provider,
                                "current_model": model,
                                "openai_api_key": api_key,
                                "project_type": project_type,
                                "keywords": keywords
                            }
                        )
                        
                        st.session_state.project_manager.save_project(project)
                        st.success(f"Project '{project_name}' created successfully!")
                        
                        # Reset form state
                        st.session_state.form_state = {
                            'name': '',
                            'description': '',
                            'enhanced_description': ''
                        }
                        
                        time.sleep(0.1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating project: {str(e)}")
            
            with col2:
                if st.button("Generate Description", key="generate_desc_btn"):
                    with st.spinner("Generating description..."):
                        enhanced_description = generate_description_sync(
                            project_name, project_description
                        )
                        st.session_state.form_state['enhanced_description'] = enhanced_description
                        st.text_area(
                            "Enhanced Description",
                            enhanced_description,
                            height=200,
                            key="enhanced_desc_area"
                        )
                        if st.button("Use This Description", key="use_desc_btn"):
                            st.session_state.form_state['description'] = enhanced_description
                            project_description = enhanced_description
        
        # Project selection
        st.subheader("Select Project")
        projects = st.session_state.project_manager.list_projects()
        if not projects:
            st.info("No projects found. Create a new project to get started.")
        else:
            selected_project = st.selectbox(
                "Choose a project",
                projects,
                key="project_selector"
            )
            if selected_project:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Load Project", key="load_project_btn"):
                        project = st.session_state.project_manager.load_project(selected_project)
                        if project:
                            st.session_state.current_project = project
                            st.success(f"Project '{selected_project}' loaded successfully!")
                        else:
                            st.error(f"Failed to load project '{selected_project}'")
                
                with col2:
                    if st.button("Delete Project", key="delete_project_btn"):
                        try:
                            st.session_state.project_manager.delete_project(selected_project)
                            st.success(f"Project '{selected_project}' deleted successfully!")
                            if hasattr(st.session_state, 'current_project') and st.session_state.current_project.name == selected_project:
                                del st.session_state.current_project
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting project: {str(e)}")
    
    # Main content area
    if hasattr(st.session_state, 'current_project'):
        project = st.session_state.current_project
        
        # Project header with status
        st.header(f"Project: {project.name}")
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            st.metric("Agents", len(project.agents))
        with status_col2:
            st.metric("Model", project.metadata["current_model"])
        with status_col3:
            st.metric("Provider", project.metadata["model_provider"])
        
        # Project description
        with st.container():
            st.markdown("### Description")
            st.markdown(project.description)
            
            if project.metadata.get("project_type"):
                st.markdown(f"**Type:** {project.metadata['project_type']}")
            if project.metadata.get("keywords"):
                st.markdown(f"**Keywords:** {', '.join(project.metadata['keywords'])}")
        
        # Project functionality tabs
        tabs = st.tabs(["Agents", "Chat", "Network", "Settings", "Features"])
        
        with tabs[0]:  # Agents
            st.subheader("Agent Management")
            
            # Create new agent
            with st.expander("Create New Agent"):
                agent_name = st.text_input("Agent Name", key="new_agent_name")
                agent_role = st.text_input("Agent Role", key="new_agent_role")
                is_orchestrator = st.checkbox("Is Orchestrator", key="new_agent_is_orchestrator")
                
                if st.button("Create Agent", key="create_agent_btn"):
                    try:
                        config = AgentConfig(
                            name=agent_name,
                            role=agent_role,
                            is_orchestrator=is_orchestrator,
                            model=project.metadata["current_model"],
                            provider=project.metadata["model_provider"]
                        )
                        agent_class = OrchestratorAgent if is_orchestrator else Agent
                        agent = agent_class(config=config, llm=project.llm_interface)
                        project.agents[agent_name] = agent
                        st.session_state.project_manager.save_project(project)
                        st.success(f"Agent '{agent_name}' created successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating agent: {str(e)}")
            
            # List existing agents
            if project.agents:
                for name, agent in project.agents.items():
                    with st.expander(f"Agent: {name}"):
                        st.write(f"Role: {agent.role}")
                        st.write(f"Type: {'Orchestrator' if agent.config.is_orchestrator else 'Specialized'}")
                        st.write(f"Status: Active")
                        if st.button(f"Delete {name}", key=f"delete_agent_{name}_btn"):
                            del project.agents[name]
                            st.session_state.project_manager.save_project(project)
                            st.success(f"Agent '{name}' deleted!")
                            st.rerun()
        
        with tabs[1]:  # Chat
            st.subheader("Chat Interface")
            
            # Initialize chat history if not exists
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("What would you like to do?", key="chat_input"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get response from orchestrator agent
                try:
                    orchestrator = next(
                        (agent for agent in project.agents.values() if agent.config.is_orchestrator),
                        None
                    )
                    
                    if orchestrator:
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                response = orchestrator.process_message(prompt)
                                st.markdown(response)
                                st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("No orchestrator agent found. Please create one first.")
                
                except Exception as e:
                    st.error(f"Error processing message: {str(e)}")
            
            # Clear chat button
            if st.button("Clear Chat", key="clear_chat_btn"):
                st.session_state.messages = []
                st.rerun()
        
        with tabs[2]:  # Network
            st.subheader("Network Visualization")
            if not project.agents:
                st.info("No agents to visualize. Create some agents to see their communication network.")
            else:
                st.write("Network visualization will be implemented here.")
        
        with tabs[3]:  # Settings
            st.subheader("Project Settings")
            
            # Model settings
            with st.expander("Model Configuration"):
                new_provider = st.selectbox(
                    "Model Provider",
                    ["ollama", "openai"],
                    index=0 if project.metadata["model_provider"] == "ollama" else 1,
                    key="settings_model_provider"
                )
                
                if new_provider == "ollama":
                    new_model = st.selectbox(
                        "Model",
                        ["mistral", "llama2", "codellama"],
                        index=0 if project.metadata["current_model"] == "mistral" else 1,
                        key="settings_model_ollama"
                    )
                    new_api_key = None
                else:
                    new_model = st.selectbox(
                        "Model",
                        ["gpt-3.5-turbo", "gpt-4"],
                        index=0,
                        key="settings_model_openai"
                    )
                    new_api_key = st.text_input(
                        "OpenAI API Key",
                        type="password",
                        value=project.metadata.get("openai_api_key", ""),
                        key="settings_api_key"
                    )
                
                if st.button("Update Model Settings", key="update_model_btn"):
                    try:
                        project.metadata.update({
                            "model_provider": new_provider,
                            "current_model": new_model,
                            "openai_api_key": new_api_key
                        })
                        project.llm_interface = init_llm_interface(
                            model_name=new_model,
                            provider=new_provider,
                            api_key=new_api_key
                        )
                        st.session_state.project_manager.save_project(project)
                        st.success("Model settings updated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error updating model settings: {str(e)}")
            
            # Project metadata
            with st.expander("Project Metadata"):
                new_type = st.selectbox(
                    "Project Type",
                    ["AI Agent System", "Web Application", "API Service", "Data Processing", "Machine Learning", "Other"],
                    index=0,
                    key="settings_project_type"
                )
                
                new_keywords = st.multiselect(
                    "Keywords",
                    ["AI", "Machine Learning", "Web", "API", "Data Processing", "Automation", "Integration", "Security", "Analytics", "Real-time"],
                    default=project.metadata.get("keywords", []),
                    key="settings_keywords"
                )
                
                if st.button("Update Metadata", key="update_metadata_btn"):
                    try:
                        project.metadata.update({
                            "project_type": new_type,
                            "keywords": new_keywords
                        })
                        st.session_state.project_manager.save_project(project)
                        st.success("Project metadata updated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error updating metadata: {str(e)}")
        
        with tabs[4]:  # Features
            st.subheader("Feature Management")
            
            # Add new feature
            with st.expander("Add New Feature"):
                feature_name = st.text_input("Feature Name", key="new_feature_name")
                feature_description = st.text_area("Feature Description", key="new_feature_desc")
                feature_status = st.selectbox(
                    "Status",
                    ["proposed", "approved", "in_progress", "completed"],
                    key="new_feature_status"
                )
                
                if st.button("Add Feature", key="add_feature_btn"):
                    try:
                        if "features" not in project.feature_bible:
                            project.feature_bible["features"] = []
                        
                        project.feature_bible["features"].append({
                            "name": feature_name,
                            "description": feature_description,
                            "status": feature_status,
                            "created_at": datetime.now().isoformat(),
                            "version": "1.0"
                        })
                        project.feature_bible["last_updated"] = datetime.now().isoformat()
                        st.session_state.project_manager.save_project(project)
                        st.success(f"Feature '{feature_name}' added successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding feature: {str(e)}")
            
            # List existing features
            if project.feature_bible.get("features"):
                for i, feature in enumerate(project.feature_bible["features"]):
                    with st.expander(f"Feature: {feature['name']}"):
                        st.write(f"Description: {feature['description']}")
                        st.write(f"Status: {feature['status']}")
                        st.write(f"Version: {feature['version']}")
                        st.write(f"Created: {feature['created_at']}")
                        
                        # Feature actions
                        col1, col2 = st.columns(2)
                        with col1:
                            new_status = st.selectbox(
                                "Update Status",
                                ["proposed", "approved", "in_progress", "completed"],
                                index=["proposed", "approved", "in_progress", "completed"].index(feature['status']),
                                key=f"feature_{i}_status"
                            )
                        with col2:
                            if st.button(f"Update {feature['name']}", key=f"update_feature_{i}_btn"):
                                feature['status'] = new_status
                                feature['version'] = f"1.{int(feature['version'].split('.')[1]) + 1}"
                                project.feature_bible["last_updated"] = datetime.now().isoformat()
                                st.session_state.project_manager.save_project(project)
                                st.success(f"Feature '{feature['name']}' updated!")
                                st.rerun()
    else:
        st.info("Select or create a project to get started.")

if __name__ == "__main__":
    main() 