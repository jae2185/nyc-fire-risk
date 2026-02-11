with open("app.py", "r") as f:
    content = f.read()

OLD = '''                        st.caption("Do zip codes flagged as high-risk actually have more fires later?")
                        tier_thresholds = [0.75, 0.50, 0.25]'''

NEW = '''                        st.caption("Do zip codes flagged as high-risk actually have more fires later?")
                        try:
                            tier_thresholds = [0.75, 0.50, 0.25]'''

# Find the one in the cloud block (after line 1600)
idx = content.find(OLD, 80000)  # skip past the main block
if idx >= 0:
    content = content[:idx] + NEW + content[idx + len(OLD):]
else:
    # Only one occurrence
    content = content.replace(OLD, NEW, 1)

# Close the try/except after the tier dataframe
OLD_END = '''                        st.dataframe(tier_display, width="stretch")

                    # ── Rolling CV from cache ──'''

NEW_END = '''                        st.dataframe(tier_display, width="stretch")
                        except Exception as e:
                            st.warning(f"Tier chart error: {e}")

                    # ── Rolling CV from cache ──'''

content = content.replace(OLD_END, NEW_END)

with open("app.py", "w") as f:
    f.write(content)

import subprocess
r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                   capture_output=True, text=True)
if r.returncode == 0:
    print("[OK] Tier chart wrapped in try/except ✅")
else:
    print(f"[ERROR] {r.stderr}")

