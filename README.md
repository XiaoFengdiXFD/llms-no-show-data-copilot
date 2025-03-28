# Purpose
This is used for analyzing project data (no-show) by utilizng LLMs. 

# General Steps
## Models Download
Deploy LLMs to local machine -- The needs for compilying with HIPAA and PHI privacy. Download Hugging Face models. Run `download_models.py`. (Note: before actual downloads, set your Hugging Face toake and Hugging Face model name).

## Test Models
Run `simple_inference.py`

## Integrate GUI
Forked from text-generation-webui
See: `https://github.com/oobabooga/text-generation-webui`