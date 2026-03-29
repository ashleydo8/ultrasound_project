import sys
import os

deps_path = "/Users/ashleydo/ultrasound_project/dependencies"

if deps_path not in sys.path:
    sys.path.append(deps_path)

# Now try the import
import pyclariuscast
print("Success! Clarius module loaded.")