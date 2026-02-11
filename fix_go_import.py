with open("app.py", "r") as f:
    content = f.read()

OLD = "import numpy as np"
NEW = "import numpy as np\nimport plotly.graph_objects as go"

content = content.replace(OLD, NEW, 1)

with open("app.py", "w") as f:
    f.write(content)

import subprocess
r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                   capture_output=True, text=True)
if r.returncode == 0:
    print("[OK] Added top-level go import ✅")
else:
    print(f"[ERROR] {r.stderr}")

