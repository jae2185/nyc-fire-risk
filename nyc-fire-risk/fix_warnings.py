"""
1. Replace use_container_width with width parameter
2. Add help tooltips to key metrics
"""

with open("app.py", "r") as f:
    content = f.read()

# ═══════════════════════════════════════════════════════════════
# TASK 1: Fix use_container_width deprecation warnings
# ═══════════════════════════════════════════════════════════════

# For plotly_chart calls
content = content.replace(
    'st.plotly_chart(fig, use_container_width=True)',
    'st.plotly_chart(fig, width="stretch")'
)

# For dataframe calls — use_container_width=True
content = content.replace('use_container_width=True', 'width="stretch"')

print("[OK] Replaced use_container_width with width='stretch'")

# ═══════════════════════════════════════════════════════════════
# TASK 2: Add help tooltips to header metrics
# ═══════════════════════════════════════════════════════════════

# Find the header metrics section
# Looking for the top-level metrics: Zones Analyzed, Structural Fires, Critical Zones, Model R²

OLD_HEADER = content  # we'll do targeted replacements

# Pattern: st.metric("Label", value) → st.metric("Label", value, help="...")
# Need to find the specific metrics. Let me check what they look like.

with open("app.py", "w") as f:
    f.write(content)

import subprocess
r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                   capture_output=True, text=True)
if r.returncode == 0:
    print("[OK] Syntax check passed ✅")
else:
    print(f"[ERROR] {r.stderr}")

