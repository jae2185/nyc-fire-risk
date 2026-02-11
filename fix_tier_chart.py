with open("app.py", "r") as f:
    content = f.read()

OLD = """                        test_zip_features = engineer_features_by_zip(test_features) if "structural_fires" not in test_features.columns else test_features
                        tier_thresholds = [0.75, 0.50, 0.25]
                        tier_labels_map = ["Critical", "High", "Moderate", "Low"]"""

NEW = """                        tier_thresholds = [0.75, 0.50, 0.25]
                        tier_labels_map = ["Critical", "High", "Moderate", "Low"]"""

# Replace BOTH occurrences (main and cloud block)
content = content.replace(OLD, NEW)

with open("app.py", "w") as f:
    f.write(content)

import subprocess
r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                   capture_output=True, text=True)
if r.returncode == 0:
    print("[OK] Removed unnecessary engineer_features_by_zip call ✅")
else:
    print(f"[ERROR] {r.stderr}")

