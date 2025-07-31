#!/usr/bin/env python3
"""
Script to run an evaluation against the RAG system API.
"""
import requests
import json
import argparse
import time
from pathlib import Path

def run_evaluation(file_path, api_url="http://localhost:8000"):
    """
    Run an evaluation using the specified request file.
    
    Args:
        file_path: Path to the evaluation request JSON file
        api_url: Base URL of the API
    """
    # Load the evaluation request
    with open(file_path, 'r') as f:
        request_data = json.load(f)
    
    # Start the evaluation
    print("Starting evaluation...")
    response = requests.post(
        f"{api_url}/evaluation/evaluate",
        json=request_data
    )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    evaluation_id = result.get("evaluation_id")
    print(f"Evaluation started with ID: {evaluation_id}")
    
    # Poll for evaluation results
    print("Waiting for evaluation to complete...")
    max_attempts = 30
    for attempt in range(max_attempts):
        time.sleep(2)  # Wait 2 seconds between checks
        
        status_response = requests.get(f"{api_url}/evaluation/evaluate/{evaluation_id}")
        status = status_response.json()
        
        if status.get("status") == "completed":
            print("\nEvaluation completed!")
            report = status.get("report")
            
            # Save the report to a file
            report_file = Path(f"evaluation_report_{evaluation_id}.md")
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"Report saved to: {report_file}")
            print("\nReport Summary:")
            print("-" * 50)
            
            # Print first 15 lines of the report as a preview
            lines = report.split("\n")
            preview_lines = min(15, len(lines))
            for i in range(preview_lines):
                print(lines[i])
            
            if len(lines) > preview_lines:
                print("...")
                print(f"See full report in {report_file}")
            
            return
        
        print(".", end="", flush=True)
    
    print("\nEvaluation is taking longer than expected.")
    print(f"You can check the status later using: GET {api_url}/evaluation/evaluate/{evaluation_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an evaluation against the RAG system API")
    parser.add_argument("file", help="Path to the evaluation request JSON file")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    
    args = parser.parse_args()
    run_evaluation(args.file, args.url)
