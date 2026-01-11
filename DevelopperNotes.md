# This is notes for me to remember stuff about the project
# and document decisions made during development.

## Run Environment
- source .venv/Scipts/activate
- CUDA torch package should be installed in venv. Uninstall torch and torchvision from venv and then run:
  - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121