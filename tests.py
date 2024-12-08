import unittest
import asyncio
from pathlib import Path
import json
import shutil
import logging
from datetime import datetime
from typing import Dict, Any
import traceback
import sys
from logging.handlers import QueueHandler
from queue import Queue
import threading

from app import Project, ProjectManager
from agent_factory import Agent, AgentConfig, AgentFactory, OrchestratorAgent
from llm_interface import LLMInterface

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Queue for capturing logs
log_queue = Queue()
queue_handler = QueueHandler(log_queue)
logger.addHandler(queue_handler)

class TestCoreFeaturesAgentSystem(unittest.TestCase):
    """Tests for Core System Features - Agent System"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - runs once before all tests"""
        logger.info(f"Starting test class: {cls.__name__}")
        cls.start_time = datetime.now()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests - runs once after all tests"""
        duration = (datetime.now() - cls.start_time).total_seconds()
        logger.info(f"Completed test class: {cls.__name__} in {duration:.2f}s")
    
    def setUp(self):
        """Set up each test"""
        self.test_name = self._testMethodName
        logger.info(f"Starting test: {self.test_name}")
        self.test_start_time = datetime.now()
        
        self.test_project_dir = Path("test_projects")
        self.test_project_dir.mkdir(exist_ok=True)
        self.project_manager = ProjectManager(str(self.test_project_dir))
        self.llm = LLMInterface()
        logger.info("Test setup completed")

    def tearDown(self):
        """Clean up after each test"""
        if self.test_project_dir.exists():
            shutil.rmtree(self.test_project_dir)
        
        duration = (datetime.now() - self.test_start_time).total_seconds()
        logger.info(f"Completed test: {self.test_name} in {duration:.2f}s")
        logger.info("Test cleanup completed")

    def test_local_llm_integration(self):
        """Test local LLM integration through Ollama"""
        try:
            logger.info("Testing LLM connection")
            result = asyncio.run(self.llm.verify_connection())
            self.assertTrue(result)
            logger.info("LLM integration test passed")
        except Exception as e:
            logger.error(f"LLM integration test failed: {str(e)}\n{traceback.format_exc()}")
            raise

    def test_dynamic_agent_creation(self):
        """Test dynamic agent creation and management"""
        try:
            logger.info("Starting agent creation test")
            
            # Test basic agent creation
            logger.debug("Creating test agent config")
            config = AgentConfig(
                name="test_agent",
                description="Test agent",
                capabilities=["test"],
                system_prompt="You are a test agent"
            )
            logger.debug(f"Created agent config: {config.dict()}")
            
            factory = AgentFactory()
            logger.debug("Created agent factory")
            
            agent = factory.create_agent(config)
            logger.debug(f"Agent created: {agent.config.dict()}")
            
            self.assertIsInstance(agent, Agent)
            self.assertEqual(agent.config.name, "test_agent")
            
            # Test agent validation
            logger.debug("Testing agent validation")
            is_healthy, _ = agent.config.perform_health_check()
            self.assertTrue(is_healthy)
            
            # Test agent capabilities
            logger.debug("Testing agent capabilities")
            self.assertEqual(agent.config.capabilities, ["test"])
            
            logger.info("Agent creation test passed")
        except Exception as e:
            logger.error(f"Agent creation test failed: {str(e)}\n{traceback.format_exc()}")
            raise

    def test_agent_health_monitoring(self):
        """Test agent health monitoring"""
        try:
            logger.info("Starting agent health monitoring test")
            
            # Create test agent
            config = AgentConfig(
                name="test_agent",
                description="Test agent",
                capabilities=["test"],
                system_prompt="You are a test agent"
            )
            logger.debug(f"Testing health check for config: {config.dict()}")
            
            # Test initial health check
            is_healthy, message = config.perform_health_check()
            logger.debug(f"Initial health check result: healthy={is_healthy}, message={message}")
            
            self.assertTrue(is_healthy)
            self.assertEqual(config.status, "healthy")
            
            # Test health check after modification
            config.error_count = 5
            is_healthy, message = config.perform_health_check()
            logger.debug(f"Health check after errors result: healthy={is_healthy}, message={message}")
            
            logger.info("Agent health monitoring test passed")
        except Exception as e:
            logger.error(f"Agent health monitoring test failed: {str(e)}\n{traceback.format_exc()}")
            raise

    def test_inter_agent_communication(self):
        """Test inter-agent communication"""
        try:
            logger.info("Starting inter-agent communication test")
            
            # Create test project and orchestrator
            logger.debug("Creating test project")
            project = self.project_manager.create_project("test_project", "Test project")
            project.initialize_orchestrator()
            logger.debug("Project and orchestrator initialized")
            
            # Create test agents
            logger.debug("Creating test agents")
            agent1_config = AgentConfig(
                name="agent1",
                description="Test agent 1",
                capabilities=["test"],
                system_prompt="You are test agent 1"
            )
            agent2_config = AgentConfig(
                name="agent2",
                description="Test agent 2",
                capabilities=["test"],
                system_prompt="You are test agent 2"
            )
            logger.debug("Agent configs created")
            
            # Add agents to project
            agent1 = project.agent_factory.create_agent(agent1_config)
            agent2 = project.agent_factory.create_agent(agent2_config)
            project.add_agent(agent1)
            project.add_agent(agent2)
            logger.debug("Agents created and added to project")
            
            # Test message passing
            test_message = "Test task requiring delegation"
            logger.debug(f"Sending test message: {test_message}")
            
            response = asyncio.run(project.process_task(test_message))
            logger.debug(f"Received response: {response}")
            
            self.assertIsNotNone(response)
            logger.info("Inter-agent communication test passed")
        except Exception as e:
            logger.error(f"Inter-agent communication test failed: {str(e)}\n{traceback.format_exc()}")
            raise

class TestModelIntegration(unittest.TestCase):
    """Tests for Model Integration features"""
    
    def setUp(self):
        self.llm = LLMInterface()

    def test_model_switching(self):
        """Test model switching capability"""
        # Test Mistral (default)
        self.assertEqual(self.llm.model_name, "mistral")
        
        # Test switching to different Ollama model
        new_llm = LLMInterface(model_name="llama2")
        self.assertEqual(new_llm.model_name, "llama2")
        
        # Test OpenAI integration (if API key available)
        if "OPENAI_API_KEY" in os.environ:
            openai_llm = LLMInterface(
                model_name="gpt-3.5-turbo",
                provider="openai",
                api_key=os.environ["OPENAI_API_KEY"]
            )
            self.assertEqual(openai_llm.provider, "openai")

    def test_model_health_monitoring(self):
        """Test model health monitoring"""
        is_healthy = asyncio.run(self.llm.verify_connection())
        self.assertTrue(is_healthy)

class TestProjectManagement(unittest.TestCase):
    """Tests for Project Management features"""
    
    def setUp(self):
        self.test_project_dir = Path("test_projects")
        self.test_project_dir.mkdir(exist_ok=True)
        self.project_manager = ProjectManager(str(self.test_project_dir))

    def tearDown(self):
        if self.test_project_dir.exists():
            shutil.rmtree(self.test_project_dir)

    def test_multiple_project_support(self):
        """Test multiple project support"""
        project1 = self.project_manager.create_project("project1", "Test project 1")
        project2 = self.project_manager.create_project("project2", "Test project 2")
        
        projects = self.project_manager.get_projects()
        self.assertEqual(len(projects), 2)
        self.assertIn("project1", projects)
        self.assertIn("project2", projects)

    def test_project_metadata(self):
        """Test project metadata tracking"""
        project = self.project_manager.create_project("test_project", "Test project")
        self.assertIsInstance(project.created_at, datetime)
        self.assertIsInstance(project.metadata, dict)

    def test_project_versioning(self):
        """Test project versioning"""
        project = self.project_manager.create_project("test_project", "Test project")
        original_version = project.feature_bible["version"]
        
        # Simulate a feature change
        asyncio.run(project.propose_feature_change("test_agent", "Add new feature"))
        self.assertNotEqual(project.feature_bible["version"], original_version)

class TestFeatureManagement(unittest.TestCase):
    """Tests for Feature Management"""
    
    def setUp(self):
        self.test_project_dir = Path("test_projects")
        self.test_project_dir.mkdir(exist_ok=True)
        self.project_manager = ProjectManager(str(self.test_project_dir))
        self.project = self.project_manager.create_project("test_project", "Test project")
        self.project.initialize_orchestrator()

    def tearDown(self):
        if self.test_project_dir.exists():
            shutil.rmtree(self.test_project_dir)

    def test_feature_proposal_system(self):
        """Test feature change proposal system"""
        proposal = "Add new test feature"
        response = asyncio.run(self.project.propose_feature_change("test_agent", proposal))
        self.assertIsNotNone(response)
        self.assertTrue(len(self.project.feature_bible["pending_changes"]) > 0)

    def test_change_approval_workflow(self):
        """Test change approval workflow"""
        # Add a test change
        asyncio.run(self.project.propose_feature_change("test_agent", "Test change"))
        
        # Test approval
        success = asyncio.run(self.project.approve_feature_change(0))
        self.assertTrue(success)
        self.assertEqual(len(self.project.feature_bible["pending_changes"]), 0)

class TestUserInterface(unittest.TestCase):
    """Tests for User Interface features"""
    
    def setUp(self):
        self.llm = LLMInterface()

    def test_ai_description_generation(self):
        """Test AI-enhanced project description generation"""
        description = asyncio.run(self.llm.generate_markdown_description(
            project_name="Test Project",
            project_type="Web Application",
            keywords=["test", "automation"]
        ))
        self.assertIsInstance(description, str)
        self.assertTrue(description.startswith("#"))

    def test_description_enhancement(self):
        """Test description enhancement capability"""
        initial_description = "# Test Project\nBasic test project"
        enhanced = asyncio.run(self.llm.enhance_description(initial_description))
        self.assertIsInstance(enhanced, str)
        self.assertNotEqual(enhanced, initial_description)

class TestAgentCapabilities(unittest.TestCase):
    """Tests for Agent Capabilities"""
    
    def setUp(self):
        self.test_project_dir = Path("test_projects")
        self.test_project_dir.mkdir(exist_ok=True)
        self.project_manager = ProjectManager(str(self.test_project_dir))
        self.project = self.project_manager.create_project("test_project", "Test project")
        self.project.initialize_orchestrator()

    def tearDown(self):
        if self.test_project_dir.exists():
            shutil.rmtree(self.test_project_dir)

    def test_task_delegation(self):
        """Test orchestrator task delegation"""
        task = "Test task requiring delegation"
        response = asyncio.run(self.project.process_task(task))
        self.assertIsNotNone(response)

    def test_specialized_agent_generation(self):
        """Test specialized agent generation"""
        description = "A project requiring data processing and API integration"
        prompt = f"""Based on this project description, list specialized AI agents needed.
        Description: {description}"""
        
        response = asyncio.run(self.project.orchestrator.execute_task(prompt))
        self.assertIsNotNone(response)

if __name__ == '__main__':
    unittest.main() 