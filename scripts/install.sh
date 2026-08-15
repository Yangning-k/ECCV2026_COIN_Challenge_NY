mkdir coin_env && cd $_ && python -m venv . && source bin/activate && cd .. 
pip install retrying flask attrs gymnasium colorama accelerate transformers==4.43.1 Pillow opencv-python dotenv qwen-vl-utils huggingface_hub google-genai openai
echo "Remember to export your GEMINI_API_KEY login token if you want to use it as the oracle. Check the README for instructions."