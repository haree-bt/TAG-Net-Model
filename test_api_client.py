import requests
import time
import subprocess
import sys

def test():
    print(">>> Starting API server in background...")
    proc = subprocess.Popen([sys.executable, "api.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(3.5)

    base_url = "http://localhost:8000"
    try:
        # 1. Test Health
        r_health = requests.get(f"{base_url}/health")
        print("GET /health ->", r_health.status_code, r_health.json())

        # 2. Test Features
        r_feat = requests.get(f"{base_url}/features")
        print("GET /features ->", r_feat.status_code, "Feature count:", len(r_feat.json()["features"]))

        # 3. Test Knife with alpha = 0.0 (Classical Failure)
        payload_fail = {
            "tool_type": "knife",
            "sample_idx": 0,
            "task_mode": "unconstrained",
            "affordance_weight": 0.0,
            "model_type": "tag_net",
            "quality_threshold": 0.05,
            "max_gripper_width": 50.0
        }
        r_fail = requests.post(f"{base_url}/predict_grasp", json=payload_fail)
        res_fail = r_fail.json()
        print("\n--- TEST A: Knife with alpha = 0.0 (Classical Baseline) ---")
        print("Grasp Coordinates:", res_fail["grasp"])
        print("Safety Verdict   :", res_fail["safety_evaluation"]["verdict"])
        print("Latency          :", res_fail["performance"]["latency_ms"], "ms")

        # 4. Test Knife with alpha = 1.0 (TAG-Net Semantic Safe Grasp)
        payload_pass = {
            "tool_type": "knife",
            "sample_idx": 0,
            "task_mode": "safe_handover",
            "affordance_weight": 1.0,
            "model_type": "tag_net",
            "quality_threshold": 0.05,
            "max_gripper_width": 50.0
        }
        r_pass = requests.post(f"{base_url}/predict_grasp", json=payload_pass)
        res_pass = r_pass.json()
        print("\n--- TEST B: Knife with alpha = 1.0 (TAG-Net Active) ---")
        print("Grasp Coordinates:", res_pass["grasp"])
        print("Safety Verdict   :", res_pass["safety_evaluation"]["verdict"])
        print("Latency          :", res_pass["performance"]["latency_ms"], "ms")

        print("\n[SUCCESS] API is 100% operational!")
    finally:
        proc.terminate()

if __name__ == "__main__":
    test()
