import os
import base64
import requests
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GitHubAPI:
    def __init__(self, token: str):
        """Initialize GitHub API with token validation."""
        if not token or not isinstance(token, str) or len(token) < 30:
            raise ValueError("Invalid GitHub token provided")
            
        self.token = token
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
        
        # Verify token
        self._verify_token()

    def _verify_token(self) -> None:
        """Verify that the token is valid by making a test API call."""
        try:
            response = requests.get(f"{self.base_url}/user", headers=self.headers)
            if response.status_code != 200:
                raise ValueError(f"Invalid token or API error: {response.text}")
            logger.info("GitHub token verified successfully")
        except Exception as e:
            raise ValueError(f"Failed to verify GitHub token: {str(e)}")

    def create_repository(self, name: str, description: str = "", private: bool = False) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Create a new GitHub repository with enhanced error handling."""
        try:
            # Validate repository name
            if not name or not isinstance(name, str) or ' ' in name:
                raise ValueError("Invalid repository name")
                
            url = f"{self.base_url}/user/repos"
            data = {
                "name": name,
                "description": description,
                "private": private,
                "auto_init": True,
                "gitignore_template": "Python",
                "license_template": "mit"
            }
            
            logger.info(f"Creating repository: {name}")
            response = requests.post(url, headers=self.headers, json=data)
            
            if response.status_code == 201:
                logger.info(f"Successfully created repository: {name}")
                return True, response.json()
            else:
                error_msg = f"Failed to create repository. Status: {response.status_code}, Response: {response.text}"
                logger.error(error_msg)
                return False, {"error": error_msg}
                
        except Exception as e:
            error_msg = f"Error creating repository: {str(e)}"
            logger.error(error_msg)
            return False, {"error": error_msg}

    def upload_file(self, repo: str, path: str, content: str, message: str = "Add file") -> Tuple[bool, str]:
        """Upload a file to the repository with enhanced error handling."""
        try:
            url = f"{self.base_url}/repos/{repo}/contents/{path}"
            logger.info(f"Uploading file: {path}")
            
            # Encode content to base64
            content_bytes = content.encode('utf-8')
            content_base64 = base64.b64encode(content_bytes).decode('utf-8')
            
            data = {
                "message": message,
                "content": content_base64
            }
            
            response = requests.put(url, headers=self.headers, json=data)
            
            if response.status_code in [201, 200]:
                logger.info(f"Successfully uploaded file: {path}")
                return True, "Success"
            else:
                error_msg = f"Failed to upload file. Status: {response.status_code}, Response: {response.text}"
                logger.error(error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"Error uploading file: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def upload_directory(self, repo: str, local_path: Path, base_path: str = "") -> Tuple[bool, Dict[str, Any]]:
        """Upload an entire directory to the repository with detailed status tracking."""
        try:
            logger.info(f"Starting upload of directory: {local_path}")
            results = {
                "success": True,
                "total_files": 0,
                "uploaded_files": 0,
                "failed_files": 0,
                "skipped_files": 0,
                "errors": []
            }
            
            # Count total files
            results["total_files"] = sum(1 for _ in local_path.rglob('*') if _.is_file() 
                                       and "__pycache__" not in str(_) 
                                       and not str(_).endswith('.pyc'))
            
            for item in local_path.iterdir():
                if item.is_file():
                    # Skip __pycache__ and .pyc files
                    if "__pycache__" in str(item) or item.suffix == ".pyc":
                        results["skipped_files"] += 1
                        logger.debug(f"Skipping file: {item}")
                        continue
                        
                    # Read file content
                    try:
                        with open(item, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except Exception as e:
                        error_msg = f"Error reading file {item}: {str(e)}"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)
                        results["failed_files"] += 1
                        results["success"] = False
                        continue
                    
                    # Create relative path
                    relative_path = str(item.relative_to(local_path))
                    if base_path:
                        relative_path = f"{base_path}/{relative_path}"
                    
                    # Upload file
                    success, message = self.upload_file(repo, relative_path, content)
                    if success:
                        results["uploaded_files"] += 1
                    else:
                        results["failed_files"] += 1
                        results["errors"].append(f"Failed to upload {relative_path}: {message}")
                        results["success"] = False
                    
                    logger.info(f"Progress: {results['uploaded_files']}/{results['total_files']} files processed")
                        
                elif item.is_dir():
                    # Skip __pycache__ directories
                    if "__pycache__" in str(item):
                        logger.debug(f"Skipping directory: {item}")
                        continue
                        
                    # Create relative path for subdirectory
                    relative_path = str(item.relative_to(local_path))
                    if base_path:
                        relative_path = f"{base_path}/{relative_path}"
                    
                    logger.info(f"Processing subdirectory: {relative_path}")
                    # Recursively upload subdirectory
                    subdir_success, subdir_results = self.upload_directory(repo, item, relative_path)
                    
                    # Update results
                    results["total_files"] += subdir_results["total_files"]
                    results["uploaded_files"] += subdir_results["uploaded_files"]
                    results["failed_files"] += subdir_results["failed_files"]
                    results["skipped_files"] += subdir_results["skipped_files"]
                    results["errors"].extend(subdir_results["errors"])
                    results["success"] = results["success"] and subdir_success
            
            if results["success"]:
                logger.info("Successfully uploaded all files")
            else:
                logger.warning("Some files failed to upload")
                
            return results["success"], results
            
        except Exception as e:
            error_msg = f"Error uploading directory: {str(e)}"
            logger.error(error_msg)
            return False, {
                "success": False,
                "error": error_msg,
                "total_files": 0,
                "uploaded_files": 0,
                "failed_files": 0,
                "skipped_files": 0,
                "errors": [error_msg]
            }

def push_to_github(token: str, repo_name: str, workspace_path: Path, description: str = "") -> Tuple[bool, Dict[str, Any]]:
    """Push the entire project to GitHub with detailed status reporting."""
    try:
        # Initialize GitHub API
        github = GitHubAPI(token)
        
        # Create repository
        repo_success, repo_info = github.create_repository(repo_name, description)
        if not repo_success:
            return False, repo_info
        
        # Get repository full name (username/repo)
        repo_full_name = repo_info['full_name']
        
        # Upload all files
        upload_success, upload_results = github.upload_directory(repo_full_name, workspace_path)
        
        results = {
            "success": upload_success,
            "repository": repo_info,
            "upload_results": upload_results
        }
        
        return upload_success, results
        
    except Exception as e:
        error_msg = f"Error pushing to GitHub: {str(e)}"
        logger.error(error_msg)
        return False, {"error": error_msg} 