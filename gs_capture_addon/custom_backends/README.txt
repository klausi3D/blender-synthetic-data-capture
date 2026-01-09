Custom Training Backends
========================

Place YAML or JSON configuration files in this folder to add custom training
backends to GS Capture.

Files:
- example_backend.yaml - Example configuration in YAML format (requires PyYAML)
- example_backend.json - Example configuration in JSON format (no dependencies)

How to Create a Custom Backend:
1. Copy example_backend.yaml or example_backend.json
2. Rename it to match your backend (e.g., my_trainer.yaml)
3. Edit the configuration to match your training framework
4. In Blender, use the "Reload Custom Backends" button or restart Blender

Configuration Fields:
- id: Unique identifier for your backend
- name: Display name in the UI
- description: Brief description
- website: URL to the project page
- detection: How to check if the backend is installed
- install_path: Where to find the installation
- environment: Python/conda environment settings
- data_format: Required data structure
- command: How to build the training command
- output_parsing: Regex patterns to parse training output
- output_files: Where to find the trained model
- install_instructions: Help text for users

Notes:
- YAML files require PyYAML to be installed in Blender's Python
- JSON files work without any additional dependencies
- Backend IDs must be unique across all backends
- Test your regex patterns before deploying

For more information, see the addon documentation.
