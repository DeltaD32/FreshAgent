import os
from pathlib import Path
import argparse
from github_utils import push_to_github
import logging
import json
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_results(results: Dict[str, Any]) -> str:
    """Format the upload results into a readable string."""
    if "error" in results:
        return f"Error: {results['error']}"
        
    upload_results = results.get("upload_results", {})
    return f"""
Upload Summary:
--------------
Total Files: {upload_results.get('total_files', 0)}
Successfully Uploaded: {upload_results.get('uploaded_files', 0)}
Failed: {upload_results.get('failed_files', 0)}
Skipped: {upload_results.get('skipped_files', 0)}

Repository URL: https://github.com/{results.get('repository', {}).get('full_name', 'unknown')}

Errors:
-------
{chr(10).join(upload_results.get('errors', []))}
"""

def main():
    parser = argparse.ArgumentParser(description='Push FreshAgent project to GitHub')
    parser.add_argument('--token', help='GitHub API token', required=True)
    parser.add_argument('--repo', help='Repository name', default='FreshAgent')
    parser.add_argument('--description', help='Repository description', 
                       default='A powerful local LLM-based agent system for creating and managing AI agents.')
    parser.add_argument('--private', help='Make repository private', action='store_true')
    args = parser.parse_args()

    try:
        # Validate token
        if not args.token or len(args.token) < 30:
            raise ValueError("Invalid GitHub token. Please provide a valid token.")

        # Get workspace path
        workspace_path = Path(__file__).parent

        # Verify workspace
        if not workspace_path.exists():
            raise ValueError(f"Workspace path does not exist: {workspace_path}")

        required_files = ['app.py', 'agent.py', 'llm_interface.py', 'requirements.txt']
        missing_files = [f for f in required_files if not (workspace_path / f).exists()]
        if missing_files:
            raise ValueError(f"Missing required files: {', '.join(missing_files)}")

        # Push to GitHub
        logger.info(f"Pushing project to GitHub repository: {args.repo}")
        success, results = push_to_github(
            token=args.token,
            repo_name=args.repo,
            workspace_path=workspace_path,
            description=args.description
        )

        if success:
            logger.info("Successfully pushed project to GitHub!")
            logger.info(format_results(results))
        else:
            logger.error("Failed to push project to GitHub")
            logger.error(format_results(results))
            exit(1)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 