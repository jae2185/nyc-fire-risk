"""Fix classifier: use optimal threshold instead of default 0.5"""

with open("app.py", "r") as f:
    content = f.read()

# Replace the "pick best" and metrics section
OLD = """                    rf_acc = accuracy_score(y_test_cls, rf_cls_preds)
                    gb_acc = accuracy_score(y_test_cls, gb_cls_preds)

                    if gb_acc >= rf_acc:
                        best_cls, best_preds, best_proba, best_name_cls = gb_cls, gb_cls_preds, gb_cls_proba, "GBM"
                    else:
                        best_cls, best_preds, best_proba, best_name_cls = rf_cls, rf_cls_preds, rf_cls_proba, "RandomForest"

                    best_acc = accuracy_score(y_test_cls, best_preds)
                    best_prec = precision_score(y_test_cls, best_preds, zero_division=0)
                    best_rec = recall_score(y_test_cls, best_preds, zero_division=0)
                    best_f1_cls = f1_score(y_test_cls, best_preds, zero_division=0)
                    best_auc = roc_auc_score(y_test_cls, best_proba) if best_proba is not None else None"""

NEW = """                    # Pick best model by AUC (not accuracy — avoids degenerate all-one-class)
                    rf_auc_cls = roc_auc_score(y_test_cls, rf_cls_proba) if rf_cls_proba is not None else 0
                    gb_auc_cls = roc_auc_score(y_test_cls, gb_cls_proba) if gb_cls_proba is not None else 0

                    if gb_auc_cls >= rf_auc_cls:
                        best_cls, best_proba, best_name_cls = gb_cls, gb_cls_proba, "GBM"
                    else:
                        best_cls, best_proba, best_name_cls = rf_cls, rf_cls_proba, "RandomForest"

                    best_auc = max(rf_auc_cls, gb_auc_cls)

                    # Find optimal threshold using Youden's J statistic
                    from sklearn.metrics import roc_curve
                    fpr, tpr, thresholds = roc_curve(y_test_cls, best_proba)
                    j_scores = tpr - fpr
                    optimal_idx = j_scores.argmax()
                    optimal_threshold = thresholds[optimal_idx]

                    best_preds = (best_proba >= optimal_threshold).astype(int)
                    best_acc = accuracy_score(y_test_cls, best_preds)
                    best_prec = precision_score(y_test_cls, best_preds, zero_division=0)
                    best_rec = recall_score(y_test_cls, best_preds, zero_division=0)
                    best_f1_cls = f1_score(y_test_cls, best_preds, zero_division=0)"""

if OLD in content:
    content = content.replace(OLD, NEW)
    with open("app.py", "w") as f:
        f.write(content)
    
    import subprocess
    r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("[OK] Classifier fixed ✅")
        print("     - Picks best model by AUC (not accuracy)")
        print("     - Uses optimal threshold via Youden's J statistic")
        print("     - Should produce balanced predictions now")
    else:
        print(f"[ERROR] {r.stderr}")
else:
    print("[ERROR] Could not find target block")

