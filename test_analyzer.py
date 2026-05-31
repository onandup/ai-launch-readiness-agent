import yaml
from serving_analyzer import ServingAnalyzer

def test_preset(filepath):
    print(f"\n--- Testing File: {filepath} ---")
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    
    result = ServingAnalyzer.analyze_workload(data)
    
    print(f"Model: {result['inputs']['model']}")
    print(f"Overall Status: {result['overall_status']} - {result['overall_status_text']}")
    print(f"Workload Classification: {result['classification']['category']}")
    print(f"Estimated ITL: {round(result['derived']['estimated_itl_ms'], 1)} ms/token")
    print(f"Estimated Concurrency: {round(result['derived']['estimated_concurrency'], 2)}")
    
    print("\nDiagnoses:")
    for diag in result['diagnoses']:
        print(f"  [{diag['severity']}] {diag['icon']} {diag['title']}")
        print(f"    Evidence: {diag['evidence']}")
        
    print("\nTop Recommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"  {i}. {rec['name']} (CLI: `{rec['cli_arg']}`)")
        print(f"     Pros: {rec['tradeoffs']['pros']}")
        
    print("\nOptimized Launch Command:")
    print(result['optimized_launch_command'])

if __name__ == "__main__":
    test_preset("sample_inputs/kv_cache_pressure.yaml")
    test_preset("sample_inputs/prefill_heavy.yaml")
    test_preset("sample_inputs/decode_heavy.yaml")
    test_preset("sample_inputs/llama_70b_kv_pressure.yaml")
    test_preset("sample_inputs/decode_heavy_low_util.yaml")
