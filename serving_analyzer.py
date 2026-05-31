"""
vLLM Doctor - serving_analyzer.py
Analytical engine that parses model serving metrics, computes derived characteristics,
and diagnoses inference bottlenecks using heuristics tailored to modern LLM serving engines (vLLM).
"""

import math
from typing import Dict, Any, List, Tuple


class ServingAnalyzer:
    """
    Analyzes model serving metrics to classify workloads, diagnose bottlenecks,
    and generate highly optimized vLLM engine configurations.
    """

    @staticmethod
    def validate_inputs(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates input metrics dictionary, applies reasonable defaults,
        and sanitizes values to prevent divide-by-zero errors.
        """
        sanitized = {}
        
        # String metrics
        sanitized["model"] = str(metrics.get("model", "unknown-model"))
        sanitized["gpu_type"] = str(metrics.get("gpu_type", "A100-SXM4-80GB"))
        sanitized["traffic_pattern"] = str(metrics.get("traffic_pattern", "constant")).lower()
        
        # Numeric metrics with sanitization (avoiding negative or zero values where problematic)
        sanitized["num_gpus"] = max(1, int(metrics.get("num_gpus", 1)))
        sanitized["qps"] = max(0.0, float(metrics.get("qps", 0.0)))
        sanitized["avg_prompt_tokens"] = max(1, int(metrics.get("avg_prompt_tokens", 1)))
        sanitized["avg_output_tokens"] = max(1, int(metrics.get("avg_output_tokens", 1)))
        sanitized["ttft_p95_sec"] = max(0.001, float(metrics.get("ttft_p95_sec", 0.1)))
        sanitized["e2e_latency_p95_sec"] = max(sanitized["ttft_p95_sec"] + 0.001, float(metrics.get("e2e_latency_p95_sec", 1.0)))
        sanitized["gpu_utilization_pct"] = min(100.0, max(0.0, float(metrics.get("gpu_utilization_pct", 0.0))))
        sanitized["kv_cache_usage_pct"] = min(100.0, max(0.0, float(metrics.get("kv_cache_usage_pct", 0.0))))
        
        return sanitized

    @staticmethod
    def compute_derived_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes derived secondary serving metrics based on validated inputs.
        """
        derived = {}
        
        # Prompt to Decode Ratio (Token context split)
        derived["prompt_to_decode_ratio"] = m["avg_prompt_tokens"] / max(1.0, m["avg_output_tokens"])
        
        # Inter-Token Latency (ITL) in milliseconds
        # Formula: E2E = TTFT + (output_tokens - 1) * ITL -> ITL = (E2E - TTFT) / (output_tokens - 1)
        decode_steps = max(1, m["avg_output_tokens"] - 1)
        total_decode_time_sec = max(0.0, m["e2e_latency_p95_sec"] - m["ttft_p95_sec"])
        derived["estimated_itl_ms"] = (total_decode_time_sec * 1000.0) / decode_steps
        
        # Active System Concurrency (Little's Law)
        # N = QPS * E2E Latency
        derived["estimated_concurrency"] = m["qps"] * m["e2e_latency_p95_sec"]
        
        # Prompt and Output Token Percentages of a single request
        total_req_tokens = m["avg_prompt_tokens"] + m["avg_output_tokens"]
        derived["prompt_token_pct"] = (m["avg_prompt_tokens"] / total_req_tokens) * 100.0
        derived["output_token_pct"] = (m["avg_output_tokens"] / total_req_tokens) * 100.0
        
        # Aggregate Token Throughput (Tokens per second total, and per GPU)
        derived["total_throughput_tps"] = m["qps"] * total_req_tokens
        derived["generation_throughput_tps"] = m["qps"] * m["avg_output_tokens"]
        derived["prefill_throughput_tps"] = m["qps"] * m["avg_prompt_tokens"]
        derived["throughput_per_gpu_tps"] = derived["total_throughput_tps"] / m["num_gpus"]
        
        return derived

    @staticmethod
    def classify_workload(m: Dict[str, Any], d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies the workload profile based on the token ratio and system load.
        """
        ratio = d["prompt_to_decode_ratio"]
        qps = m["qps"]
        gpu_util = m["gpu_utilization_pct"]
        
        if qps < 0.1 or (gpu_util < 10.0 and d["estimated_concurrency"] < 0.2):
            category = "Idle / Under-utilized"
            description = "The system is receiving negligible traffic or experiencing extreme resource starvation from the client-side."
            color = "gray"
        elif ratio > 5.0:
            category = "Prefill-Heavy"
            description = (
                f"Workload is heavily dominated by large prompt sizes (avg prompt: {m['avg_prompt_tokens']} tokens, "
                f"avg output: {m['avg_output_tokens']} tokens). Typical of summarization, document QA, RAG, or classification tasks."
            )
            color = "#8a2be2"  # Purple
        elif ratio < 0.2:
            category = "Decode-Heavy"
            description = (
                f"Workload is heavily dominated by long sequence generation (avg prompt: {m['avg_prompt_tokens']} tokens, "
                f"avg output: {m['avg_output_tokens']} tokens). Typical of code generation, creative writing, or step-by-step reasoning agents."
            )
            color = "#00ced1"  # Cyan
        else:
            category = "Balanced Conversational"
            description = (
                f"Workload exhibits a balanced split between prefill and decode phases (avg prompt: {m['avg_prompt_tokens']} tokens, "
                f"avg output: {m['avg_output_tokens']} tokens). Typical of conversational chat agents and multi-turn assistants."
            )
            color = "#4caf50"  # Green
            
        return {
            "category": category,
            "description": description,
            "color": color,
            "ratio": ratio
        }

    @staticmethod
    def diagnose_bottlenecks(m: Dict[str, Any], d: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Diagnoses specific performance bottlenecks in the model serving system using expert heuristics.
        Returns a list of diagnosed bottlenecks sorted by severity (Critical -> Warning -> Healthy).
        """
        diagnoses = []
        
        # 1. KV Cache Saturation
        if m["kv_cache_usage_pct"] >= 90.0:
            diagnoses.append({
                "severity": "CRITICAL",
                "title": "KV Cache Saturation / Memory Exhaustion",
                "evidence": f"KV cache usage is at {m['kv_cache_usage_pct']}%",
                "description": (
                    "The GPU memory allocated for key-value caching is almost completely full. This prevents the "
                    "vLLM engine from scheduling new requests, forcing them to queue in CPU memory or causing "
                    "active sequences to be preempted (swapped out) to CPU memory. This drastically spikes latency "
                    "and causes inter-token latency (ITL) jitter."
                ),
                "metric_impacted": "kv_cache_usage_pct",
                "icon": "🔴"
            })
        elif m["kv_cache_usage_pct"] >= 75.0:
            diagnoses.append({
                "severity": "WARNING",
                "title": "KV Cache Pressure",
                "evidence": f"KV cache usage is at {m['kv_cache_usage_pct']}%",
                "description": (
                    "The KV cache is approaching saturation. While currently functional, the system is highly vulnerable "
                    "to sudden bursts of traffic or conversational history growth, which will trigger active sequence swapping."
                ),
                "metric_impacted": "kv_cache_usage_pct",
                "icon": "🟡"
            })

        # 2. Severe Prefill Queuing / High TTFT
        # High TTFT (> 1.5 seconds) relative to workload
        if m["ttft_p95_sec"] > 2.0:
            diagnoses.append({
                "severity": "CRITICAL",
                "title": "Severe Prefill Starvation / High TTFT",
                "evidence": f"P95 TTFT is {m['ttft_p95_sec']}s",
                "description": (
                    "Users are waiting over 2 seconds just to receive their first token. This is typically caused by "
                    "incoming prefill requests queuing up because the GPUs are busy completing long-running decode phases, "
                    "or because massive prompt batches are saturating the GPU compute units simultaneously."
                ),
                "metric_impacted": "ttft_p95_sec",
                "icon": "🔴"
            })
        elif m["ttft_p95_sec"] > 1.0 or (m["ttft_p95_sec"] / m["e2e_latency_p95_sec"] > 0.4 and m["avg_prompt_tokens"] > 1000):
            diagnoses.append({
                "severity": "WARNING",
                "title": "Elevated Time-to-First-Token (TTFT)",
                "evidence": f"P95 TTFT is {m['ttft_p95_sec']}s ({round((m['ttft_p95_sec'] / m['e2e_latency_p95_sec']) * 100, 1)}% of total latency)",
                "description": (
                    "TTFT is elevated, comprising a substantial portion of the end-to-end response time. Prefill phases "
                    "are blocking decode cycles, or the queue is starting to form under load."
                ),
                "metric_impacted": "ttft_p95_sec",
                "icon": "🟡"
            })

        # 3. GPU Compute Saturation
        if m["gpu_utilization_pct"] >= 92.0 and m["qps"] >= 0.5:
            diagnoses.append({
                "severity": "WARNING",
                "title": "GPU Compute Bound (High Utilization)",
                "evidence": f"GPU utilization is at {m['gpu_utilization_pct']}% under QPS {m['qps']}",
                "description": (
                    "The GPU execution units are fully saturated. While this represents excellent hardware ROI and "
                    "high throughput, it means the system has zero head-room to accommodate traffic spikes. Any increase "
                    "in QPS will cause immediate backlog accumulation and exponential queuing latency spikes."
                ),
                "metric_impacted": "gpu_utilization_pct",
                "icon": "🟡"
            })

        # 4. Low Hardware Saturation / Inefficient Serving
        if m["gpu_utilization_pct"] < 30.0 and m["qps"] > 0.1:
            if d["estimated_itl_ms"] > 80.0 and d["prompt_to_decode_ratio"] < 0.2:
                # High ITL but low GPU utilization is classic Memory-Bandwidth limit in Decode-Heavy task with small batch size
                diagnoses.append({
                    "severity": "CRITICAL",
                    "title": "Memory-Bandwidth Bound (Low Batch Size Starvation)",
                    "evidence": f"GPU util is {m['gpu_utilization_pct']}% but ITL is high ({round(d['estimated_itl_ms'], 1)} ms)",
                    "description": (
                        "A classic LLM serving bottleneck: the GPU is heavily bottlenecked by memory bandwidth rather than "
                        "compute. Sequential token generation (decode) requires reloading the entire model weights from GPU "
                        "High Bandwidth Memory (HBM) to SRAM for every single token. With low batch sizes, the GPU cores sit idle "
                        "waiting for weights, leading to low utilization and high inter-token latency (ITL)."
                    ),
                    "metric_impacted": "gpu_utilization_pct",
                    "icon": "🔴"
                })
            else:
                diagnoses.append({
                    "severity": "WARNING",
                    "title": "GPU Under-utilization / Capacity Waste",
                    "evidence": f"GPU utilization is {m['gpu_utilization_pct']}% under QPS {m['qps']}",
                    "description": (
                        "The serving nodes are significantly under-provisioned relative to the demand, or the client is not "
                        "sending enough concurrent requests to saturate the model serving batch slots. This represents "
                        "unnecessary spending and oversized hardware infrastructure."
                    ),
                    "metric_impacted": "gpu_utilization_pct",
                    "icon": "🟡"
                })

        # 5. Communication Overhead (Tensor Parallel overhead)
        if m["num_gpus"] > 1 and d["estimated_itl_ms"] > 100.0:
            # High ITL on multi-GPU usually points to tensor parallel sync delay, especially over PCIe vs NVLink
            diagnoses.append({
                "severity": "WARNING",
                "title": "Cross-GPU Interconnect Bottleneck",
                "evidence": f"Running on {m['num_gpus']} GPUs with high ITL ({round(d['estimated_itl_ms'], 1)} ms)",
                "description": (
                    "With Tensor Parallelism (TP) > 1, GPUs must perform All-Reduce synchronization multiple times per "
                    "transformer block. If the server does not have NVLink (e.g. standard PCIe slots), the inter-GPU "
                    "communication latency can quickly dominate, stalling the cores and spiking the generation times."
                ),
                "metric_impacted": "num_gpus",
                "icon": "🟡"
            })

        # 6. Default Healthy State
        if not diagnoses:
            diagnoses.append({
                "severity": "HEALTHY",
                "title": "Balanced & Healthy Serving State",
                "evidence": f"GPU Util: {m['gpu_utilization_pct']}%, KV Cache: {m['kv_cache_usage_pct']}%, TTFT: {m['ttft_p95_sec']}s",
                "description": (
                    "The model serving instance is operating within optimal parameters. There are no signs of KV cache "
                    "exhaustion, severe queuing delays, or hardware starvation. The workload matches the hardware profile nicely."
                ),
                "metric_impacted": "gpu_utilization_pct",
                "icon": "🟢"
            })

        return diagnoses

    @staticmethod
    def generate_recommendations(m: Dict[str, Any], d: Dict[str, Any], diagnoses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates action-oriented tuning recommendations for the vLLM engine based on the diagnosed bottlenecks.
        """
        recs = []
        is_critical_kv = any(diag["title"].startswith("KV Cache Saturation") for diag in diagnoses)
        is_kv_pressure = any("KV Cache" in diag["title"] for diag in diagnoses)
        is_prefill_starve = any("Prefill Starvation" in diag["title"] for diag in diagnoses) or any("Elevated Time-to-First-Token" in diag["title"] for diag in diagnoses)
        is_mem_bandwidth_bound = any("Memory-Bandwidth Bound" in diag["title"] for diag in diagnoses)
        is_underutil = any("Under-utilization" in diag["title"] for diag in diagnoses)
        is_interconnect = any("Interconnect" in diag["title"] for diag in diagnoses)
        
        # Recommendation 1: Chunked Prefill
        # Excellent for prefill-heavy workloads, high TTFT, or high KV cache utilization
        if is_prefill_starve or d["prompt_to_decode_ratio"] > 4.0:
            recs.append({
                "name": "Enable Chunked Prefill",
                "cli_arg": "--enable-chunked-prefill",
                "description": (
                    "By default, vLLM executes prefill requests entirely before allowing decodes. Under high concurrency, "
                    "large prompts block ongoing generations, causing spikes in inter-token latency. Enabling chunked "
                    "prefill splits large prompt prefill operations into chunks, interleaving them with decode phases "
                    "for exceptionally smooth generation latency and vastly reduced prefill-side queuing."
                ),
                "impact": "Decreases P95 ITL and prevents prefill-starvation; may slightly increase individual prefill execution times.",
                "tradeoff_key": "chunked_prefill"
            })

        # Recommendation 2: Automatic Prefix Caching
        # Ideal when prompts are long or workload is RAG/multi-turn chat
        if m["avg_prompt_tokens"] >= 1024 or m["traffic_pattern"] in ["bursty", "spike"]:
            recs.append({
                "name": "Enable Automatic Prefix Caching",
                "cli_arg": "--enable-prefix-caching",
                "description": (
                    "Automatically caches and reuses the KV cache of prompt prefixes (system prompts, standard documents, "
                    "or historical turns in a conversation). If multiple users hit the same model with matching contexts, "
                    "the prefill phase is skipped entirely, turning a several-second latency step into near-zero milliseconds."
                ),
                "impact": "Drastically slashes TTFT (up to 95% reduction) and saves substantial KV cache memory for repetitive queries.",
                "tradeoff_key": "prefix_caching"
            })

        # Recommendation 3: Adjust Max Num Seqs (vLLM Max Batch Size)
        if is_critical_kv:
            recs.append({
                "name": "Reduce Maximum Sequences (Prevent KV OOM / Swapping)",
                "cli_arg": "--max-num-seqs 128",
                "description": (
                    "The current high KV cache utilization suggests vLLM is attempting to process too many parallel "
                    "sequences at once, causing cache thrashing or CPU swapping. Reducing `--max-num-seqs` from the default "
                    "256 down to 128 (or 64) establishes a strict boundary on concurrently active batches, capping the "
                    "maximum memory allocations."
                ),
                "impact": "Protects against GPU OOMs and eliminates high-latency swapping; may lower absolute peak throughput under extreme concurrency.",
                "tradeoff_key": "max_num_seqs"
            })
        elif is_mem_bandwidth_bound or is_underutil:
            # Low util, decode heavy -> need to increase active batch size to saturate memory bandwidth!
            recs.append({
                "name": "Increase Max Num Seqs (Saturate Memory Bandwidth)",
                "cli_arg": "--max-num-seqs 512",
                "description": (
                    "The GPU is running well below capacity during sequential token generation because the active batch size is too small. "
                    "Increasing `--max-num-seqs` from the default 256 to 512 enables vLLM to group more requests into continuous "
                    "batches. This amortizes the model parameter loading costs across far more sequences in each forward pass."
                ),
                "impact": "Significantly boosts tokens/sec throughput and increases GPU utilization; slightly increases KV cache requirements.",
                "tradeoff_key": "max_num_seqs_increase"
            })

        # Recommendation 4: Optimize GPU Memory Utilization
        if is_kv_pressure and not is_critical_kv:
            recs.append({
                "name": "Increase GPU Memory Allocation for KV Cache",
                "cli_arg": "--gpu-memory-utilization 0.95",
                "description": (
                    "By default, vLLM reserves 90% of available GPU memory for model weights and the KV cache. If this instance "
                    "is dedicated solely to serving this single model, increasing `--gpu-memory-utilization` to 0.95 "
                    "allocates an extra 5% of memory directly to the KV cache pool, providing substantial breathing room."
                ),
                "impact": "Expands KV cache capacity, delaying or avoiding scheduling bottlenecks; increases the risk of OOM if supplementary allocations occur.",
                "tradeoff_key": "gpu_mem_util"
            })

        # Recommendation 5: Quantization (FP8 / AWQ)
        # Highly relevant for large models or high KV pressure
        is_large_model = "70B" in m["model"].upper() or "70b" in m["model"] or "large" in m["model"].lower()
        if is_kv_pressure or is_large_model or m["gpu_type"].startswith("H100") or m["gpu_type"].startswith("L4"):
            # FP8 is natively supported on Ada Lovelace (L4, L40S) and Hopper (H100)
            rec_text = "Enable FP8 Quantization"
            rec_arg = "--quantization fp8"
            if "A100" in m["gpu_type"] or "A10G" in m["gpu_type"]:
                rec_text = "Enable AWQ Weights Quantization / FP8 KV Cache"
                rec_arg = "--quantization awq --kv-cache-dtype fp8"
                
            recs.append({
                "name": rec_text,
                "cli_arg": rec_arg,
                "description": (
                    "Compacting weights or KV cache activations to 8-bit floating point (FP8) drastically reduces the "
                    "memory footprint of both the model parameters and active sequence contexts. This allows a single "
                    "GPU to hold twice as much data, boosting max batch capacity and reducing memory-bus saturation."
                ),
                "impact": "Halves model memory usage, doubles maximum throughput, and increases token generation speeds; may introduce negligible precision degradation.",
                "tradeoff_key": "quantization"
            })

        # Recommendation 6: Speculative Decoding
        # Ideal for Decode-Heavy, low-concurrency, or where user needs ultra-fast response for individual clients
        if d["prompt_to_decode_ratio"] < 0.2 and m["qps"] <= 3.0:
            draft_model = "ibm-fms/llama3-13b-instruct-decoding-assistant" if "70B" in m["model"].upper() else "meta-llama/Meta-Llama-3-8B-Instruct"
            recs.append({
                "name": "Deploy Speculative Decoding",
                "cli_arg": f"--speculative-model {draft_model} --num-speculative-tokens 5",
                "description": (
                    "Decode execution is highly sequential and memory-bound. Speculative decoding uses a smaller, hyper-fast "
                    "draft model to guess the next 4-5 tokens in parallel, which the larger main model validates in a single "
                    "GPU forward pass. This breaks the sequential token-at-a-time bottleneck."
                ),
                "impact": "Increases single-sequence decoding speed (up to 1.5x - 2x faster) and slashes ITL; increases GPU memory usage and consumes extra compute.",
                "tradeoff_key": "speculative_decoding"
            })

        # Recommendation 7: Interconnect Scaling Adjustments
        if is_interconnect:
            recs.append({
                "name": "Deconsolidate Tensor Parallelism / Deploy Pipeline Parallelism",
                "cli_arg": f"--tensor-parallel-size {max(1, m['num_gpus'] // 2)} --pipeline-parallel-size 2",
                "description": (
                    "When interconnect lanes (like PCIe) lack sufficient NVLink bandwidth, Tensor Parallelism causes substantial "
                    "inter-GPU overhead as weights are split vertically. Combining Tensor Parallelism (TP) with Pipeline Parallelism "
                    "(PP) allows layers to be split horizontally across nodes. This drastically reduces the sync barriers "
                    "required during a forward pass."
                ),
                "impact": "Decreases inter-token sync delays on servers without NVLink; introduces minor pipeline bubble overhead.",
                "tradeoff_key": "pp_vs_tp"
            })
            
        # Default fallback if healthy
        if not recs:
            recs.append({
                "name": "Scale Down Replicas to Reduce Costs",
                "cli_arg": "Scale nodes down",
                "description": (
                    "The system is running highly efficiently and has significant head-room. Consider reducing the number of "
                    "active nodes, downscaling to a cost-efficient GPU (like L4 or L40S), or setting up auto-scaling rules."
                ),
                "impact": "Reduces infrastructure bills linearly while maintaining performance SLAs.",
                "tradeoff_key": "scale_down"
            })

        return recs

    @staticmethod
    def get_tradeoff_details(key: str) -> Dict[str, Any]:
        """
        Returns structured tradeoff explanations for different optimization mechanics.
        """
        tradeoffs = {
            "chunked_prefill": {
                "pros": ["Provides highly stable, predictable Inter-Token Latency (ITL)", "Eliminates high-concurrency prefill queuing", "Prevents generation jitter"],
                "cons": ["Slightly increases individual prefill execution times (higher base TTFT for isolated requests)", "Adds small vLLM scheduling overhead"],
                "complexity": "Low (Simple flag `--enable-chunked-prefill`)"
            },
            "prefix_caching": {
                "pros": ["Slashes TTFT to near-zero for repeating prompts", "Drastically cuts down redundant GPU memory usage"],
                "cons": ["Only effective if there are redundant, overlapping prefixes", "Slightly consumes host CPU memory to manage metadata structures"],
                "complexity": "Low (Simple flag `--enable-prefix-caching`)"
            },
            "max_num_seqs": {
                "pros": ["Establishes safe memory upper bounds", "Ensures stable performance and guarantees no KV cache swapping"],
                "cons": ["Caps maximum parallel concurrency", "May increase request queuing at the server boundary under peak traffic"],
                "complexity": "Medium (Requires tuning based on expected max context lengths)"
            },
            "max_num_seqs_increase": {
                "pros": ["Significantly boosts multi-user throughput", "Increases GPU hardware ROI"],
                "cons": ["Increases peak KV cache size requirements", "Slightly elevates aggregate ITL as batches grow larger"],
                "complexity": "Medium (Requires validating context boundaries)"
            },
            "gpu_mem_util": {
                "pros": ["Increases maximum possible active batch sizes", "Helps fit longer context sequences into high-speed memory"],
                "cons": ["Increases risk of out-of-memory (OOM) errors during complex CUDA operations or unexpected auxiliary memory allocations"],
                "complexity": "Low (Flag `--gpu-memory-utilization`)"
            },
            "quantization": {
                "pros": ["Halves model and KV cache memory requirements", "Saturates memory-bandwidth bound decode cycles", "Slashes ITL by up to 40%"],
                "cons": ["Can cause minor quality loss on subtle reasoning, mathematics, or extremely sensitive formatting instructions", "Requires modern GPU support (e.g. FP8 is best on Hopper/Ada Lovelace)"],
                "complexity": "Medium (Requires validating model output quality)"
            },
            "speculative_decoding": {
                "pros": ["Accelerates individual generation speeds by 1.5x - 2.0x", "Significantly improves real-time chat feel"],
                "cons": ["Increases memory requirements (must load both target and draft models)", "Consumes additional GPU compute power", "Ineffective if draft model acceptance rate is poor (< 60%)"],
                "complexity": "High (Requires selecting and deploying a matching fast draft model)"
            },
            "pp_vs_tp": {
                "pros": ["Substantially reduces cross-GPU synchronization over slower PCIe interfaces", "Avoids networking-bound bottlenecks"],
                "cons": ["Introduces pipeline bubbles (some GPUs stand idle while waiting for layers to complete horizontal passes)", "Slightly more complex serving topology"],
                "complexity": "High (Requires tuning TP vs PP boundaries)"
            },
            "scale_down": {
                "pros": ["Unlocks immediate cost savings (often 50% or more)", "Saves idle power and cuts infrastructure waste"],
                "cons": ["Reduces absolute performance capacity, leaving system more vulnerable to sudden traffic spikes"],
                "complexity": "Medium (Requires setting up scale-to-zero or auto-scaling groups)"
            }
        }
        
        return tradeoffs.get(key, {
            "pros": ["Improves system serving performance"],
            "cons": ["May require adjusting configuration files"],
            "complexity": "Low"
        })

    @classmethod
    def analyze_workload(cls, raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main orchestration endpoint. Validates, derives, classifies, diagnoses,
        and generates comprehensive configuration recommendations.
        """
        m = cls.validate_inputs(raw_metrics)
        d = cls.compute_derived_metrics(m)
        w = cls.classify_workload(m, d)
        diagnoses = cls.diagnose_bottlenecks(m, d)
        recs = cls.generate_recommendations(m, d, diagnoses)
        
        # Inject detailed tradeoff records into each recommendation
        for r in recs:
            trade_key = r.get("tradeoff_key", "default")
            r["tradeoffs"] = cls.get_tradeoff_details(trade_key)
            
        # Determine overall status of the system
        critical_count = sum(1 for diag in diagnoses if diag["severity"] == "CRITICAL")
        warning_count = sum(1 for diag in diagnoses if diag["severity"] == "WARNING")
        
        if critical_count > 0:
            overall_status = "CRITICAL"
            overall_status_text = "System exhibits critical performance bottleneck"
            color = "#f44336"  # Red
        elif warning_count > 0:
            overall_status = "WARNING"
            overall_status_text = "System is operating with suboptimal latency or headroom risks"
            color = "#ff9800"  # Orange
        else:
            overall_status = "HEALTHY"
            overall_status_text = "System is running optimally with healthy hardware margins"
            color = "#4caf50"  # Green

        # Executive Summary Generation
        summary = cls.generate_executive_summary(m, d, diagnoses, w)
        
        # Build copyable launch command
        tp_size = m["num_gpus"]
        launch_command = f"python3 -m vllm.entrypoints.openai.api_server \\\n  --model {m['model']} \\\n  --tensor-parallel-size {tp_size}"
        
        for r in recs:
            if r["cli_arg"] and not r["cli_arg"].startswith("Scale") and not r["cli_arg"].startswith("Deconsolidate"):
                launch_command += f" \\\n  {r['cli_arg']}"
                
        return {
            "inputs": m,
            "derived": d,
            "classification": w,
            "diagnoses": diagnoses,
            "recommendations": recs,
            "overall_status": overall_status,
            "overall_status_text": overall_status_text,
            "overall_color": color,
            "executive_summary": summary,
            "optimized_launch_command": launch_command
        }

    @staticmethod
    def generate_executive_summary(m: Dict[str, Any], d: Dict[str, Any], diagnoses: List[Dict[str, Any]], workload: Dict[str, Any]) -> str:
        """
        Generates clinical-grade executive summary ("doctor's diagnosis").
        """
        model_name = m["model"].split("/")[-1]
        gpu_count_type = f"{m['num_gpus']}x {m['gpu_type']}"
        workload_desc = workload["category"]
        
        # Extract primary bottlenecks
        critical_names = [diag["title"] for diag in diagnoses if diag["severity"] == "CRITICAL"]
        warning_names = [diag["title"] for diag in diagnoses if diag["severity"] == "WARNING"]
        
        summary_lines = [
            f"**Headroom AI Infrastructure Analysis Report for {model_name}**",
            f"- **Hardware Environment:** {gpu_count_type}",
            f"- **Workload Characterization:** {workload_desc} profile executing at {m['qps']} QPS",
        ]
        
        if critical_names:
            summary_lines.append(f"- **Primary Pathological Finding:** 🔴 **{', '.join(critical_names)}** was identified as a critical inhibitor of performance.")
        elif warning_names:
            summary_lines.append(f"- **Secondary Pathological Finding:** 🟡 **{', '.join(warning_names)}** is creating latency penalties or capacity risks.")
        else:
            summary_lines.append("- **Primary Finding:** 🟢 The serving instance is in highly robust health with optimal utilization profiles.")

        # Latency diagnostics
        summary_lines.append(
            f"- **Latency Metrics:** P95 TTFT is **{round(m['ttft_p95_sec'], 2)}s** (contributing {round((m['ttft_p95_sec']/m['e2e_latency_p95_sec'])*100, 1)}% of total response times). "
            f"The estimated Inter-Token Latency (ITL) is **{round(d['estimated_itl_ms'], 1)} ms/token**."
        )
        
        # Concurrency & utilization
        summary_lines.append(
            f"- **Load Profile:** Average concurrency of **{round(d['estimated_concurrency'], 2)}** requests. "
            f"GPU core utilization is **{m['gpu_utilization_pct']}%**, with **{m['kv_cache_usage_pct']}%** KV cache capacity utilized."
        )

        # High-level prognosis
        if critical_names:
            summary_lines.append(
                "**Prognosis & Action Plan:** Immediate intervention recommended. The active bottlenecks are bloating user-facing latency. "
                "Implementing chunked prefill or enabling quantization will stabilize queue cycles, protect against Out-Of-Memory (OOM) failures, "
                "and align resource demand back within high-speed GPU boundaries."
            )
        elif warning_names:
            summary_lines.append(
                "**Prognosis & Action Plan:** Recommended preventive care. While the system is holding, it is vulnerable or running inefficiently. "
                "Applying minor engine parameter updates (such as adjusting max sequence limits or prefix caching) will reclaim lost performance margins."
            )
        else:
            summary_lines.append(
                "**Prognosis & Action Plan:** Maintain current setup. The system is operating safely within bounds. Excellent deployment sizing."
            )

        return "\n".join(summary_lines)
