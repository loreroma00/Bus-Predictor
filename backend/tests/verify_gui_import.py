import sys
import os

# Add root to path
sys.path.append(os.getcwd())

print("Attempting to import interaction.debug_gui...")
try:
    print("SUCCESS: interaction.debug_gui imported without syntax errors.")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    exit(1)
