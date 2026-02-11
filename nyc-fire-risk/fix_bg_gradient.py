with open("app.py", "r") as f:
    content = f.read()

# Fix 1: Rankings table (line ~230)
OLD1 = """            ranked[display_cols].style.format({
                "structural_fire_rate": "{:.1%}",
                "risk_score": "{:.3f}",
                "predicted_fires": "{:.0f}",
            }).background_gradient(
                subset=["risk_score"], cmap="YlOrRd"
            ),
            use_container_width=True,
            height=500,"""

NEW1 = """            ranked[display_cols],
            use_container_width=True,
            height=500,"""

# Fix 2: Constituent zips (line ~368)
OLD2 = """                            constituent_zips[["zip_code", "structural_fires", "risk_score", "risk_label"]]
                            .sort_values("risk_score", ascending=False)
                            .style.format({"risk_score": "{:.3f}"})
                            .background_gradient(subset=["risk_score"], cmap="YlOrRd"),
                            use_container_width=True,
                            height=250,"""

NEW2 = """                            constituent_zips[["zip_code", "structural_fires", "risk_score", "risk_label"]]
                            .sort_values("risk_score", ascending=False),
                            use_container_width=True,
                            height=250,"""

content = content.replace(OLD1, NEW1)
content = content.replace(OLD2, NEW2)

with open("app.py", "w") as f:
    f.write(content)

import subprocess
r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                   capture_output=True, text=True)
print("[OK] Removed background_gradient calls" if r.returncode == 0 else f"[ERROR] {r.stderr}")

