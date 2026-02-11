with open("app.py", "r") as f:
    content = f.read()

# Fix plotly_chart calls — revert to use_container_width=True
content = content.replace(
    'st.plotly_chart(fig, width="stretch")',
    'st.plotly_chart(fig, use_container_width=True)'
)
content = content.replace(
    'st.plotly_chart(fig2, width="stretch")',
    'st.plotly_chart(fig2, use_container_width=True)'
)
content = content.replace(
    'st.plotly_chart(fig_cm, width="stretch")',
    'st.plotly_chart(fig_cm, use_container_width=True)'
)
content = content.replace(
    'st.plotly_chart(fig_imp_cls, width="stretch")',
    'st.plotly_chart(fig_imp_cls, use_container_width=True)'
)

with open("app.py", "w") as f:
    f.write(content)

# Verify
import subprocess
r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                   capture_output=True, text=True)
print("[OK] Fixed plotly_chart calls ✅" if r.returncode == 0 else f"[ERROR] {r.stderr}")

# Count
remaining = content.count('width="stretch"')
plotly_fixed = content.count('use_container_width=True')
print(f"     Remaining width='stretch' (dataframes only): {remaining}")
print(f"     plotly_chart with use_container_width: {plotly_fixed}")

