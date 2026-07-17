import os
import shutil
import tempfile
from pathlib import Path
import sys

# Add current directory to sys.path to allow importing llmcli
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from llmcli.tools import _repo_map, estimate_text_tokens

def test_integration_repo_map_budget_with_query():
    # 1. Setup temporary workspace inside the current directory to pass _within_workspace check
    test_dir = Path.cwd() / "test_workspace_tmp"
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Create a file with significant content
        file1 = test_dir / "large_file.txt"
        file1.write_text("This is a long string of text that should definitely exceed a budget of 1 token." * 10)
        
        # Create a second file
        file2 = test_dir / "small_file.txt"
        file2.write_text("Small")

        # 2. Call _repo_map with a tiny budget AND a query (to prevent 8x expansion)
        args = {
            "path": str(test_dir),
            "max_map_tokens": 1,  # Extremely low budget
            "query": "something"   # Non-empty query to ensure target = max_map_tokens
        }
        
        result = _repo_map(args)
        
        # 3. Validate
        if not result.get("ok"):
            print(f"Error: _repo_map failed: {result.get('error')}")
            sys.exit(1)
            
        rendered_text = result.get("result", "")
        
        print(f"Rendered text: '{rendered_text}'")
        
        # If the fix works, rendered_text should be empty because the first file 
        # exceeds the budget and we removed the 'else blocks[:1]' fallback.
        # Note: The output includes a header, so we check if the header shows 0 files.
        if "0 files" in rendered_text:
            print("SUCCESS: Budget was respected. No files rendered.")
            sys.exit(0)
        else:
            print("FAILURE: Budget was NOT respected. Content was rendered despite low budget.")
            sys.exit(1)
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_integration_repo_map_budget_with_query()
