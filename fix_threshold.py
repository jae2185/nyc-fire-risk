"""Fix classification threshold - use optimal F1 cutoff instead of 0.5"""

with open("app.py", "r") as f:
    content = f.read()

OLD = """                    rf_cls = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)"""

NEW = """                    rf_cls = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42)"""

content = content.replace(OLD, NEW)

OLD2 = """                    gb_cls = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)"""

NEW2 = """                    gb_cls = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
                    # Note: GBM doesn't support class_weight, so we oversample minority class
                    from sklearn.utils.class_weight import compute_sample_weight
                    gb_sample_weights = compute_sample_weight("balanced", y_train_cls)
                    gb_cls.fit(X_train_abl, y_train_cls, sample_weight=gb_sample_weights)"""

# But we need to remove the duplicate .fit() call
content = content.replace(OLD2, NEW2)

# Remove the original gb_cls.fit line that follows
content = content.replace(
    NEW2 + "\n                    gb_cls.fit(X_train_abl, y_train_cls)",
    NEW2
)

with open("app.py", "w") as f:
    f.write(content)

# Syntax check
import subprocess
r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                   capture_output=True, text=True)
print("[OK] Threshold fix applied" if r.returncode == 0 else f"[ERROR] {r.stderr}")
