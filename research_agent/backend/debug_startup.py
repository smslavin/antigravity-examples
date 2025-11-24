import os
import sys
import traceback

try:
    print("Importing main...")
    from main import app
    print("Main imported successfully.")
    
    # Trigger startup events if any (though we don't have explicit ones, components are init at module level)
    print("Components initialized.")
    
except Exception as e:
    print("Error during startup:")
    traceback.print_exc()
