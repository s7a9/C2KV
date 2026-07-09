MODEL="qwen3-32b/260703"
AGENT_LLM_ARGS='{"api_base":"http://localhost:32002/v1","api_key":"dummy"}'
USER_LLM_ARGS='{"api_base":"http://localhost:32001/v1","api_key":"dummy"}'
DOMAIN='airline'

cd /home/duchuheng/C2KV/tau2-bench

uv run tau2 run \
  --domain $DOMAIN \
  --agent llm_agent \
  --agent-llm 'openai/checkpoints/${MODEL}' \
  --agent-llm-args $AGENT_LLM_ARGS \
  --user-llm 'openai/checkpoints/${MODEL}' \
  --user-llm-args $USER_LLM_ARGS \
  --num-trials 1 \
  --max-concurrency 4 \
  --max-steps 30 \
  --save-to ${MODEL}/c2kv \
  --verbose-logs \
  --llm-log-mode all


uv run tau2 run \
  --domain $DOMAIN \
  --agent llm_agent \
  --agent-llm 'openai/checkpoints/${MODEL}' \
  --agent-llm-args $AGENT_LLM_ARGS \
  --user-llm 'openai/checkpoints/${MODEL}' \
  --user-llm-args $USER_LLM_ARGS \
  --num-trials 1 \
  --max-concurrency 4 \
  --max-steps 30 \
  --save-to ${MODEL}/full \
  --verbose-logs \
  --llm-log-mode all