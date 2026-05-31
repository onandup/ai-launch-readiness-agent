import os
import yaml
import asyncio
import argparse
import math
from typing import Dict, Any, Optional
from google.antigravity import Agent, LocalAgentConfig, ToolContext
from serving_analyzer import ServingAnalyzer

# ==========================================
# 🛠️ MANAGED AGENT TOOLS DEFINITION
# ==========================================

def classify_workload(
    metrics: dict,
    ctx: Optional[ToolContext] = None
) -> dict:
    """Classifies the LLM serving workload based on metric characteristics and returns derived characteristics.
    
    Args:
        metrics: Dictionary containing model serving metrics (model, gpu_type, num_gpus, qps, avg_prompt_tokens, avg_output_tokens, ttft_p95_sec, e2e_latency_p95_sec, gpu_utilization_pct, kv_cache_usage_pct, traffic_pattern).
    """
    try:
        m = ServingAnalyzer.validate_inputs(metrics)
        d = ServingAnalyzer.compute_derived_metrics(m)
        classification = ServingAnalyzer.classify_workload(m, d)
        
        res = {
            "workload_classification": classification["category"],
            "description": classification["description"],
            "color": classification["color"],
            "derived_metrics": {
                "prompt_to_decode_ratio": round(d["prompt_to_decode_ratio"], 4),
                "estimated_itl_ms": round(d["estimated_itl_ms"], 2),
                "estimated_concurrency": round(d["estimated_concurrency"], 2),
                "total_token_throughput": round(d["total_throughput_tps"], 2),
                "prompt_token_pct": round(d["prompt_token_pct"], 2),
                "output_token_pct": round(d["output_token_pct"], 2)
            }
        }
        if ctx:
            ctx.set_state("metrics", m)
            ctx.set_state("derived", d)
            ctx.set_state("classification", res)
        return res
    except Exception as e:
        return {"error": f"Error during workload classification: {e}"}


def analyze_kv_cache(
    metrics: dict,
    ctx: Optional[ToolContext] = None
) -> dict:
    """Audits KV cache memory occupancy and predicts saturation / thrashing risks.
    
    Args:
        metrics: Dictionary containing model serving metrics.
    """
    try:
        m = ServingAnalyzer.validate_inputs(metrics)
        d = ServingAnalyzer.compute_derived_metrics(m)
        diagnoses = ServingAnalyzer.diagnose_bottlenecks(m, d)
        
        kv_diagnoses = [dg for dg in diagnoses if "KV Cache" in dg["title"] or "Memory" in dg["title"] or dg["metric_impacted"] == "kv_cache_usage_pct"]
        
        status = "HEALTHY"
        if any(kd["severity"] == "CRITICAL" for kd in kv_diagnoses):
            status = "CRITICAL"
        elif any(kd["severity"] == "WARNING" for kd in kv_diagnoses):
            status = "WARNING"
            
        res = {
            "status": status,
            "kv_cache_usage_pct": m["kv_cache_usage_pct"],
            "diagnoses": [{
                "severity": kd["severity"],
                "title": kd["title"],
                "evidence": kd["evidence"],
                "description": kd["description"]
            } for kd in kv_diagnoses]
        }
        if ctx:
            ctx.set_state("kv_cache_analysis", res)
            existing = ctx.get_state("diagnoses", [])
            for kd in kv_diagnoses:
                if kd["title"] not in [e["title"] for e in existing]:
                    existing.append(kd)
            ctx.set_state("diagnoses", existing)
        return res
    except Exception as e:
        return {"error": f"Error during KV cache analysis: {e}"}


def analyze_latency(
    metrics: dict,
    ctx: Optional[ToolContext] = None
) -> dict:
    """Analyzes prompt and decode latency characteristics, mapping them to bottleneck sources.
    
    Args:
        metrics: Dictionary containing model serving metrics.
    """
    try:
        m = ServingAnalyzer.validate_inputs(metrics)
        d = ServingAnalyzer.compute_derived_metrics(m)
        diagnoses = ServingAnalyzer.diagnose_bottlenecks(m, d)
        
        latency_diagnoses = [dg for dg in diagnoses if "Latency" in dg["title"] or "TTFT" in dg["title"] or "Prefill" in dg["title"] or "Stall" in dg["title"] or dg["metric_impacted"] in ["ttft_p95_sec", "e2e_latency_p95_sec"]]
        
        status = "HEALTHY"
        if any(ld["severity"] == "CRITICAL" for ld in latency_diagnoses):
            status = "CRITICAL"
        elif any(ld["severity"] == "WARNING" for ld in latency_diagnoses):
            status = "WARNING"
            
        res = {
            "status": status,
            "ttft_p95_sec": m["ttft_p95_sec"],
            "e2e_latency_p95_sec": m["e2e_latency_p95_sec"],
            "estimated_itl_ms": round(d["estimated_itl_ms"], 2),
            "diagnoses": [{
                "severity": ld["severity"],
                "title": ld["title"],
                "evidence": ld["evidence"],
                "description": ld["description"]
            } for ld in latency_diagnoses]
        }
        if ctx:
            ctx.set_state("latency_analysis", res)
            existing = ctx.get_state("diagnoses", [])
            for ld in latency_diagnoses:
                if ld["title"] not in [e["title"] for e in existing]:
                    existing.append(ld)
            ctx.set_state("diagnoses", existing)
        return res
    except Exception as e:
        return {"error": f"Error during latency analysis: {e}"}


def analyze_gpu_utilization(
    metrics: dict,
    ctx: Optional[ToolContext] = None
) -> dict:
    """Evaluates GPU core saturation versus capacity waste and detects efficiency pathologies.
    
    Args:
        metrics: Dictionary containing model serving metrics.
    """
    try:
        m = ServingAnalyzer.validate_inputs(metrics)
        d = ServingAnalyzer.compute_derived_metrics(m)
        diagnoses = ServingAnalyzer.diagnose_bottlenecks(m, d)
        
        util_diagnoses = [dg for dg in diagnoses if "utilization" in dg["title"].lower() or "Compute" in dg["title"] or "Under-utilization" in dg["title"] or "starvation" in dg["title"].lower() or dg["metric_impacted"] == "gpu_utilization_pct"]
        
        status = "HEALTHY"
        if any(ud["severity"] == "CRITICAL" for ud in util_diagnoses):
            status = "CRITICAL"
        elif any(ud["severity"] == "WARNING" for ud in util_diagnoses):
            status = "WARNING"
            
        res = {
            "status": status,
            "gpu_utilization_pct": m["gpu_utilization_pct"],
            "diagnoses": [{
                "severity": ud["severity"],
                "title": ud["title"],
                "evidence": ud["evidence"],
                "description": ud["description"]
            } for ud in util_diagnoses]
        }
        if ctx:
            ctx.set_state("gpu_util_analysis", res)
            existing = ctx.get_state("diagnoses", [])
            for ud in util_diagnoses:
                if ud["title"] not in [e["title"] for e in existing]:
                    existing.append(ud)
            ctx.set_state("diagnoses", existing)
        return res
    except Exception as e:
        return {"error": f"Error during GPU utilization analysis: {e}"}


def recommend_optimizations(
    metrics: dict,
    analyses: Optional[dict] = None,
    ctx: Optional[ToolContext] = None
) -> dict:
    """Generates autotuned vLLM CLI parameters and profiles operational tradeoffs.
    
    Args:
        metrics: Dictionary containing model serving metrics.
        analyses: Optional dictionary containing prior analyses of the workload.
    """
    try:
        m = ServingAnalyzer.validate_inputs(metrics)
        d = ServingAnalyzer.compute_derived_metrics(m)
        diagnoses = ServingAnalyzer.diagnose_bottlenecks(m, d)
        recs = ServingAnalyzer.generate_recommendations(m, d, diagnoses)
        
        tp_size = m["num_gpus"]
        launch_command = f"python3 -m vllm.entrypoints.openai.api_server \\\n  --model {m['model']} \\\n  --tensor-parallel-size {tp_size}"
        for r in recs:
            if r["cli_arg"] and not r["cli_arg"].startswith("Scale") and not r["cli_arg"].startswith("Deconsolidate"):
                launch_command += f" \\\n  {r['cli_arg']}"
                
        tradeoffs = []
        for r in recs:
            t_details = ServingAnalyzer.get_tradeoff_details(r["tradeoff_key"])
            tradeoffs.append({
                "recommendation": r["name"],
                "cli_argument": r["cli_arg"],
                "mitigation_target": r["description"],
                "pros": t_details.get("pros", []),
                "cons": t_details.get("cons", []),
                "risk_profile": t_details.get("complexity", "Low")
            })
            
        res = {
            "optimized_vllm_command": launch_command,
            "recommendations": [{
                "name": r["name"],
                "cli_override_argument": r["cli_arg"],
                "impact": r["impact"],
                "rationale": r["description"]
            } for r in recs],
            "architectural_tradeoffs_and_risks": tradeoffs
        }
        if ctx:
            ctx.set_state("recommendations", res)
        return res
    except Exception as e:
        return {"error": f"Error generating recommendations: {e}"}


def generate_engineer_report(
    metrics: dict,
    analyses: Optional[dict] = None,
    ctx: Optional[ToolContext] = None
) -> dict:
    """Generates a structured engineering report detailing CUDA execution limits, HBM transfer constraints, and parameter rationales.
    
    Args:
        metrics: Dictionary containing model serving metrics.
        analyses: Optional dictionary containing prior analyses of the workload.
    """
    try:
        m = ServingAnalyzer.validate_inputs(metrics)
        d = ServingAnalyzer.compute_derived_metrics(m)
        diagnoses = ServingAnalyzer.diagnose_bottlenecks(m, d)
        recs = ServingAnalyzer.generate_recommendations(m, d, diagnoses)
        
        cuda_hbm_audit = []
        if m["kv_cache_usage_pct"] >= 90.0:
            cuda_hbm_audit.append(
                "CRITICAL CUDA HBM SATURATION: Active KV cache memory footprint is expanding exponentially, exhausting physical High Bandwidth Memory. "
                "The vLLM scheduler is starved of physical HBM block slots, triggering thread execution stalls and sequence swapping to host CPU memory. "
                "This causes massive PCIe bus transfer delays and spikes Inter-Token Latency."
            )
        elif m["kv_cache_usage_pct"] >= 75.0:
            cuda_hbm_audit.append(
                "MODERATE HBM CAPACITY PRESSURE: High HBM occupancy limits concurrent scheduling. "
                "Any transient traffic spikes will exhaust the active KV block pool and trigger paging/swapping overhead."
            )
            
        if d["prompt_to_decode_ratio"] > 5.0:
            cuda_hbm_audit.append(
                "COMPUTE-BOUND PREFILL PATHOLOGY: The high prompt-to-decode ratio saturates Tensor Cores with compute-bound GEMM matrix multiplications. "
                "Because vLLM prioritizes prefills, ongoing sequential decodes are starved, inflating P95 TTFT and stalling existing generation streams."
            )
        elif d["prompt_to_decode_ratio"] < 0.2:
            cuda_hbm_audit.append(
                "MEMORY-BANDWIDTH BOUND DECODE BOTTLENECK: Sequential token generation requires reloading the entire model parameter weight matrix from HBM "
                "to high-speed SRAM for every single forward pass. With low concurrency/batch sizes, the massive compute capability of the GPU is starved "
                "while waiting for memory-bus transfer times, representing severe hardware efficiency waste."
            )
            
        if m["num_gpus"] > 1 and d["estimated_itl_ms"] > 100.0:
            cuda_hbm_audit.append(
                "INTER-GPU INTERCONNECT SYNC STALLS: High multi-GPU communication latency points to tensor parallel sync delay (All-Reduce operations). "
                "If GPUs lack high-speed NVLink interconnects and communicate over PCIe, communication overhead dominates token generation cycles."
            )
            
        if not cuda_hbm_audit:
            cuda_hbm_audit.append(
                "OPTIMAL HARDWARE ALIGNMENT: Serving weights and active sequence allocations fit comfortably within HBM capacity. "
                "Thread warp scheduler execution shows well-balanced prefill/decode cycles with minimal stall times."
            )
            
        tp_size = m["num_gpus"]
        launch_command = f"python3 -m vllm.entrypoints.openai.api_server \\\n  --model {m['model']} \\\n  --tensor-parallel-size {tp_size}"
        for r in recs:
            if r["cli_arg"] and not r["cli_arg"].startswith("Scale") and not r["cli_arg"].startswith("Deconsolidate"):
                launch_command += f" \\\n  {r['cli_arg']}"
                
        res = {
            "cuda_hbm_hardware_audit": "\n\n".join(cuda_hbm_audit),
            "recommended_terminal_command": launch_command,
            "parameter_overrides_rationales": [{
                "cli_argument": r["cli_arg"],
                "impact_profile": r["impact"],
                "remediation_rationale": r["description"]
            } for r in recs]
        }
        if ctx:
            ctx.set_state("engineer_report", res)
        return res
    except Exception as e:
        return {"error": f"Error generating engineering report: {e}"}


def generate_executive_summary(
    metrics: dict,
    analyses: Optional[dict] = None,
    ctx: Optional[ToolContext] = None
) -> dict:
    """Generates a highly condensed executive health report with core action plans.
    
    Args:
        metrics: Dictionary containing model serving metrics.
        analyses: Optional dictionary containing prior analyses of the workload.
    """
    try:
        m = ServingAnalyzer.validate_inputs(metrics)
        d = ServingAnalyzer.compute_derived_metrics(m)
        diagnoses = ServingAnalyzer.diagnose_bottlenecks(m, d)
        classification = ServingAnalyzer.classify_workload(m, d)
        recs = ServingAnalyzer.generate_recommendations(m, d, diagnoses)
        
        overall_status = "HEALTHY"
        if any(dg["severity"] == "CRITICAL" for dg in diagnoses):
            overall_status = "CRITICAL"
        elif any(dg["severity"] == "WARNING" for dg in diagnoses):
            overall_status = "WARNING"
            
        severity_text = "exhibits critical performance bottlenecks that demand immediate remediation" if overall_status == "CRITICAL" else ("is operating with suboptimal latency or headroom risks" if overall_status == "WARNING" else "is operating within healthy parameters")
        
        remediation_command_args = [r["cli_arg"] for r in recs if r["cli_arg"]]
        action_plan_summary = f"Deploy the autotuned vLLM server configuration with overrides: {', '.join(remediation_command_args)}." if remediation_command_args else "No immediate changes required; system is operating optimally."
        
        core_bottleneck_description = "None"
        critical_diags = [dg for dg in diagnoses if dg["severity"] == "CRITICAL"]
        warning_diags = [dg for dg in diagnoses if dg["severity"] == "WARNING"]
        if critical_diags:
            core_bottleneck_description = f"{critical_diags[0]['title']}: {critical_diags[0]['evidence']}. {critical_diags[0]['description'][:150]}..."
        elif warning_diags:
            core_bottleneck_description = f"{warning_diags[0]['title']}: {warning_diags[0]['evidence']}. {warning_diags[0]['description'][:150]}..."
        else:
            core_bottleneck_description = "The serving pipeline is perfectly optimized and balanced for the incoming workload."
            
        res = {
            "overall_health_status": overall_status,
            "classification_category": classification["category"],
            "brief_status_assessment": f"The model serving pipeline {severity_text}.",
            "core_diagnosed_bottleneck": core_bottleneck_description,
            "high_level_action_plan": action_plan_summary
        }
        if ctx:
            ctx.set_state("executive_summary", res)
        return res
    except Exception as e:
        return {"error": f"Error generating executive summary: {e}"}


def run_what_if_analysis(
    metrics: dict,
    scenario: dict,
    ctx: Optional[ToolContext] = None
) -> dict:
    """Projects system operational health and performance under hypothetical workload adjustments.
    
    Args:
        metrics: Dictionary containing baseline model serving metrics.
        scenario: Dictionary containing scaling adjustments (qps_multiplier, prompt_length_multiplier, hardware_upgrade, add_gpus_count).
    """
    try:
        m = ServingAnalyzer.validate_inputs(metrics)
        hypothetical = m.copy()
        consequences = []
        
        # Apply QPS Multiplier
        qps_mult = float(scenario.get("qps_multiplier", 1.0))
        if qps_mult != 1.0:
            hypothetical["qps"] = m["qps"] * qps_mult
            consequences.append(f"QPS scales by {qps_mult}x from {m['qps']} to {hypothetical['qps']}.")
            if qps_mult > 1.0:
                hypothetical["kv_cache_usage_pct"] = min(100.0, m["kv_cache_usage_pct"] * math.sqrt(qps_mult))
                hypothetical["gpu_utilization_pct"] = min(100.0, m["gpu_utilization_pct"] * math.sqrt(qps_mult))
                
                if m["gpu_utilization_pct"] > 80.0 or m["kv_cache_usage_pct"] > 80.0:
                    hypothetical["ttft_p95_sec"] = m["ttft_p95_sec"] * (qps_mult ** 1.8)
                    hypothetical["e2e_latency_p95_sec"] = m["e2e_latency_p95_sec"] * (qps_mult ** 1.5)
                    consequences.append("Hardware resources were near saturation; scaling QPS is projected to cause queue accumulation and exponential latencies.")
                else:
                    hypothetical["ttft_p95_sec"] = m["ttft_p95_sec"] * qps_mult
                    hypothetical["e2e_latency_p95_sec"] = m["e2e_latency_p95_sec"] * qps_mult
                    consequences.append("System headroom can absorb initial demand, but queue lengths and active concurrency will swell.")
            else:
                hypothetical["kv_cache_usage_pct"] = max(5.0, m["kv_cache_usage_pct"] * qps_mult)
                hypothetical["gpu_utilization_pct"] = max(5.0, m["gpu_utilization_pct"] * qps_mult)
                hypothetical["ttft_p95_sec"] = max(0.01, m["ttft_p95_sec"] * qps_mult)
                hypothetical["e2e_latency_p95_sec"] = max(0.02, m["e2e_latency_p95_sec"] * qps_mult)
                consequences.append("Decreasing workload frequency frees scheduling pressure and resolves cache pressure.")
                
        # Apply Prompt Length Multiplier
        prompt_mult = float(scenario.get("prompt_length_multiplier", 1.0))
        if prompt_mult != 1.0:
            hypothetical["avg_prompt_tokens"] = int(m["avg_prompt_tokens"] * prompt_mult)
            consequences.append(f"Average prompt length scales by {prompt_mult}x from {m['avg_prompt_tokens']} to {hypothetical['avg_prompt_tokens']} tokens.")
            if prompt_mult > 1.0:
                hypothetical["ttft_p95_sec"] = m["ttft_p95_sec"] * prompt_mult
                hypothetical["e2e_latency_p95_sec"] = m["e2e_latency_p95_sec"] + (hypothetical["ttft_p95_sec"] - m["ttft_p95_sec"])
                hypothetical["kv_cache_usage_pct"] = min(100.0, m["kv_cache_usage_pct"] * math.sqrt(prompt_mult))
                consequences.append("Longer prompt sequences scale prefill compute demand, increasing TTFT queuing delays.")
            else:
                hypothetical["ttft_p95_sec"] = max(0.005, m["ttft_p95_sec"] * prompt_mult)
                hypothetical["e2e_latency_p95_sec"] = max(0.01, m["e2e_latency_p95_sec"] - (m["ttft_p95_sec"] - hypothetical["ttft_p95_sec"]))
                hypothetical["kv_cache_usage_pct"] = max(5.0, m["kv_cache_usage_pct"] * prompt_mult)
                consequences.append("Shorter prompt sequences release memory-bus and compute prefill bounds.")
                
        # Apply Hardware Upgrade
        hw_upgrade = str(scenario.get("hardware_upgrade", "")).upper()
        if hw_upgrade:
            hypothetical["gpu_type"] = hw_upgrade
            if "H100" in hw_upgrade:
                hypothetical["ttft_p95_sec"] = max(0.05, m["ttft_p95_sec"] / 3.0)
                hypothetical["gpu_utilization_pct"] = max(10.0, m["gpu_utilization_pct"] / 1.7)
                hypothetical["e2e_latency_p95_sec"] = hypothetical["ttft_p95_sec"] + max(0.05, (m["e2e_latency_p95_sec"] - m["ttft_p95_sec"]) / 1.7)
                consequences.append("Upgraded node to H100 with higher memory bandwidth, substantially accelerating prefill and decode execution speed.")
            elif "A100" in hw_upgrade and "A10G" in m["gpu_type"].upper():
                hypothetical["ttft_p95_sec"] = max(0.1, m["ttft_p95_sec"] / 2.0)
                hypothetical["e2e_latency_p95_sec"] = hypothetical["ttft_p95_sec"] + max(0.1, (m["e2e_latency_p95_sec"] - m["ttft_p95_sec"]) / 1.5)
                consequences.append("Upgraded to A100 SXM, expanding active memory pool to bypass caching pressure.")
                
        # Apply Additional GPUs
        add_gpus_val = int(scenario.get("add_gpus_count", 0))
        if add_gpus_val > 0:
            hypothetical["num_gpus"] = m["num_gpus"] + add_gpus_val
            consequences.append(f"Adding {add_gpus_val} GPUs brings the cluster nodes up to {hypothetical['num_gpus']} GPUs.")
            scale_factor = m["num_gpus"] / hypothetical["num_gpus"]
            hypothetical["kv_cache_usage_pct"] = max(5.0, m["kv_cache_usage_pct"] * scale_factor)
            hypothetical["gpu_utilization_pct"] = max(5.0, m["gpu_utilization_pct"] * scale_factor)
            consequences.append("KV cache pool bounds expanded, protecting active sessions from queue drops.")
            
        hypo_validated = ServingAnalyzer.validate_inputs(hypothetical)
        hypo_derived = ServingAnalyzer.compute_derived_metrics(hypo_validated)
        hypo_diagnoses = ServingAnalyzer.diagnose_bottlenecks(hypo_validated, hypo_derived)
        
        overall_status_hypo = "HEALTHY"
        if any(dg["severity"] == "CRITICAL" for dg in hypo_diagnoses):
            overall_status_hypo = "CRITICAL"
        elif any(dg["severity"] == "WARNING" for dg in hypo_diagnoses):
            overall_status_hypo = "WARNING"
            
        return {
            "scenario_evaluated": scenario,
            "consequences_observed": consequences,
            "baseline_metrics": m,
            "projected_metrics": hypo_validated,
            "projected_derived": {
                "prompt_to_decode_ratio": round(hypo_derived["prompt_to_decode_ratio"], 4),
                "estimated_itl_ms": round(hypo_derived["estimated_itl_ms"], 2),
                "estimated_concurrency": round(hypo_derived["estimated_concurrency"], 2),
                "total_token_throughput": round(hypo_derived["total_throughput_tps"], 2)
            },
            "projected_diagnoses": [{
                "severity": dg["severity"],
                "title": dg["title"],
                "evidence": dg["evidence"]
            } for dg in hypo_diagnoses],
            "projected_overall_status": overall_status_hypo,
            "remediation_status": f"System projected to run in a {overall_status_hypo} state under this hypothetical scenario."
        }
    except Exception as e:
        return {"error": f"Error during what-if analysis: {e}"}


# ==========================================
# 🤖 SYSTEM PERSONA DEFINITIONS
# ==========================================

VLLM_DOCTOR_SYSTEM_PROMPT = (
    "You are Headroom, the AI Infrastructure Advisor, an elite senior AI Infrastructure Engineer "
    "specializing in LLM serving efficiency. Your role is to analyze workloads, diagnose "
    "memory/compute bottlenecks, and generate optimized vLLM config/launch command recommendations.\n\n"
    "Your analysis MUST ALWAYS be fully consistent with the structured heuristics outputs provided by "
    "your specialized tools. Do NOT deviate from, alter, or ignore their findings. They are your primary "
    "source of truth.\n\n"
    "To analyze a user's workload, you must execute these tools logically:\n"
    "1. `classify_workload(metrics)`: Call this first to identify the core profile category (e.g. Prefill-Heavy, Decode-Heavy) and fetch calculated derived metrics.\n"
    "2. `analyze_kv_cache(metrics)`: Call this to audit memory allocations, caching efficiency, and detect swap thrashing.\n"
    "3. `analyze_latency(metrics)`: Call this to evaluate queuing constraints (P95 TTFT) and generation bottlenecks (ITL).\n"
    "4. `analyze_gpu_utilization(metrics)`: Call this to evaluate core saturation versus capacity waste.\n"
    "5. `recommend_optimizations(metrics, analyses)`: Call this to generate the autotuned vLLM server execution parameters and tradeoffs.\n"
    "6. `generate_engineer_report(metrics, analyses)`: Call this to compile clinical-grade architectural CUDA/HBM audits and command overrides.\n"
    "7. `generate_executive_summary(metrics, analyses)`: Call this to generate a condensed high-level operational health assessment and action plan.\n"
    "8. `run_what_if_analysis(metrics, scenario)`: Call this to run projections under scaling QPS, doubled prompt size, or hardware migrations.\n\n"
    "COOPERATION GUIDELINES:\n"
    "Always run these tools consecutively or together when presented with a serving scenario. Once you obtain the outputs, "
    "synthesize a highly polished, professional product-quality report structured EXACTLY with these sections in this order:\n\n"
    "### 1. ENGINEER REPORT\n"
    "- Provide a deep-dive, clinical-grade architectural audit of CUDA execution limits, HBM-to-SRAM memory transfer bounds, or cross-GPU interconnect bandwidth latencies.\n"
    "- Include a physical code block with the recommended optimized terminal execution command (`python3 -m vllm.entrypoints.openai.api_server ...`) and parameter rationales.\n\n"
    "### 2. EXECUTIVE SUMMARY\n"
    "- A condensed 2-3 sentence technical overview of the system's operational health, core bottleneck, and the high-level remediation action plan.\n\n"
    "### 3. EVIDENCE FROM METRICS\n"
    "- Highlight exact metric values from the input metrics as concrete pathology evidence.\n"
    "- Explicitly show how secondary derived characteristics (such as Prompt-to-Decode Token Ratio, Inter-Token Latency, and Active Concurrency) corroborate this diagnosis.\n\n"
    "### 4. RISKS AND TRADEOFFS\n"
    "- Detail the operational tradeoffs of the recommended adjustments (e.g. latency stability vs. prefill overhead, precision loss vs. memory savings, cost boundaries vs. replication throughput).\n"
    "- CRITICAL: Do NOT claim exact savings percentages or performance improvements unless backed by raw mathematical calculations. Always use ranges and explicit assumptions. Do not invent exact savings.\n\n"
    "### 5. WHAT-IF ANALYSIS\n"
    "- Provide forward-looking projections of how the system would behave under hypothetical workload adjustments:\n"
    "  * **What if QPS scales by 2x?**: Project how this affects queuing, KV cache saturation, and TTFT/E2E latency.\n"
    "  * **What if average prompt length doubles?**: Discuss prefill compute bounds vs. decode phases and possible memory swapping.\n"
    "  * **What if we upgrade hardware (e.g., A100 to H100 or adding GPUs)?**: Discuss latency amortization and scaling bottlenecks.\n\n"
    "Tone: Authoritative, expert, clinical, highly technical. Use clean, beautiful markdown."
)


# ==========================================
# 🚀 CLI ENTRYPOINT & LOOP
# ==========================================

async def chat_with_agent(query: str):
    """Initializes the Agent and performs a single chat run."""
    config = LocalAgentConfig(
        tools=[classify_workload, analyze_kv_cache, analyze_latency, analyze_gpu_utilization, recommend_optimizations, generate_engineer_report, generate_executive_summary, run_what_if_analysis],
        system_instructions=VLLM_DOCTOR_SYSTEM_PROMPT
    )
    
    async with Agent(config) as agent:
        print(f"\n🩺 [Headroom Advisor Agent] User Query: '{query}'")
        print("💡 Diagnosing workload with active heuristics tools...")
        
        response = await agent.chat(query)
        
        print("\n📝 [Compiled Staff Engineer Report]:\n" + "="*50)
        async for chunk in response:
            print(chunk, end="", flush=True)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headroom AI Infrastructure Advisor Managed Agent Console")
    parser.add_argument("query", nargs="?", help="Analytical workload query for the agent")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive chat session")
    args = parser.parse_args()
    
    if args.interactive:
        async def run_interactive():
            config = LocalAgentConfig(
                tools=[
                    classify_workload, analyze_kv_cache, analyze_latency, analyze_gpu_utilization,
                    recommend_optimizations, generate_engineer_report, generate_executive_summary,
                    run_what_if_analysis
                ],
                system_instructions=VLLM_DOCTOR_SYSTEM_PROMPT
            )
            print("🩺 Headroom Managed Advisor Agent is ready. Entering interactive loop...")
            async with Agent(config) as agent:
                await agent.run_interactive_loop()
        try:
            asyncio.run(run_interactive())
        except KeyboardInterrupt:
            print("\n👋 Exiting console.")
    elif args.query:
        asyncio.run(chat_with_agent(args.query))
    else:
        # Default run with the standard Llama 70B KV Pressure pathology
        sample_query = (
            "Diagnose the following workload metrics: model is Meta-Llama-3-70B-Instruct, running "
            "on 2 A100-SXM4-80GB GPUs. Incoming request load is at 12.0 QPS. "
            "The average prompt sequence length is 2048 tokens, average generation length is 256 tokens. "
            "Metrics audit shows P95 TTFT is 2.85 seconds, P95 E2E latency is 12.5 seconds, "
            "GPU utilization is 94.0%, KV cache usage is 98.0%, traffic is bursty."
        )
        asyncio.run(chat_with_agent(sample_query))
