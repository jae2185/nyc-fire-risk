with open("app.py", "r") as f:
    content = f.read()

OLD = '''                    if demo_cache.exists():
                        import pickle
                        with open(demo_cache, "rb") as f:
                            st.session_state[data_key] = pickle.load(f)
                        st.toast("Loaded cached validation data from disk")'''

NEW = '''                    if demo_cache.exists():
                        import pickle
                        with open(demo_cache, "rb") as f:
                            st.session_state[data_key] = pickle.load(f)
                        st.toast("Loaded cached validation data from disk")
                        _vc = st.session_state[data_key]
                        train_features = _vc["train_features"]
                        test_features = _vc["test_features"]
                        X_train_full = _vc["X_train_full"]
                        X_test_full = _vc["X_test_full"]
                        y_train = _vc["y_train"]
                        y_test = _vc["y_test"]
                        fn_full = _vc["fn_full"]'''

content = content.replace(OLD, NEW)

with open("app.py", "w") as f:
    f.write(content)

import subprocess
r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                   capture_output=True, text=True)
print("[OK] Fixed cache unpacking ✅" if r.returncode == 0 else f"[ERROR] {r.stderr}")

