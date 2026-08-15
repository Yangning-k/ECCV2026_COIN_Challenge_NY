mkdir vllm_env && cd $_ && python -m venv . && source bin/activate && cd .. 
source vllm_env/bin/activate
pip install vllm
