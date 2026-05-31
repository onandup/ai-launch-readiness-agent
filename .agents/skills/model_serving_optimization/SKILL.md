# Model Serving Optimization Skill

Use these heuristics:

- KV cache usage > 85% means memory pressure.
- GPU utilization < 50% means likely batching, scheduling, memory, or queueing inefficiency.
- avg_prompt_tokens > 4 * avg_output_tokens means prefill-heavy.
- avg_output_tokens > avg_prompt_tokens means decode-heavy.
- High TTFT usually points to prefill, queueing, batching, or cold start.
- High E2E latency with acceptable TTFT usually points to decode throughput limits.

Recommend:
- chunked prefill for prefill-heavy workloads
- continuous batching tuning for low GPU utilization
- speculative decoding for decode-heavy workloads
- KV cache tuning when memory pressure is high
- better observability when metrics are insufficient
