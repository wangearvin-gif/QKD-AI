"""B3/B5：加入相對特徵後，比較完整模型和 QBER 門檻偵測器抓到竊聽的比例，並輸出 f 曲線。"""

import argparse
from collections import defaultdict
from pathlib import Path

from train_decision_tree import (
    FEATURE_COLUMNS,
    SimpleDecisionTree,
    load_dataset,
    train_test_split,
)

def flag_rate_by_class(rows, flagged):
    """每一類 block 裡，被判成竊聽（flagged 為 True）的比例。"""
    counts = defaultdict(int)
    hits = defaultdict(int)
    for row, flag in zip(rows, flagged):
        counts[row["label"]] += 1
        if flag:
            hits[row["label"]] += 1
    return {label: hits[label] / counts[label] for label in sorted(counts)}

def overall_false_alarm(rows, flagged):
    """所有非竊聽 block 裡，被判成竊聽的比例。"""
    pairs = [(row, flag) for row, flag in zip(rows, flagged) if row["label"] != "intercept_resend"]
    return sum(flag for row, flag in pairs) / len(pairs)

# ==========================================
# 1. 讀資料與特徵工程 (新增相對基準特徵)
# ==========================================
parser = argparse.ArgumentParser(description="B3/B5：協定門檻、QBER 門檻、決策樹的竊聽偵測比較")
parser.add_argument("--input", type=Path, default=Path("data/qkd_dataset_diagnostic.csv"))
args = parser.parse_args()

raw_rows = load_dataset(args.input)

# 計算第 0 區塊的基準點擊率與累積錯誤率
baseline_click = {}
cumulative_z_err = defaultdict(int)
cumulative_x_err = defaultdict(int)
cumulative_z_total = defaultdict(int)
cumulative_x_total = defaultdict(int)

# 依 run_id 排序確保時序正確，提取新增特徵
for row in sorted(raw_rows, key=lambda x: (x.get("run_id", 0), x.get("block_idx", 0))):
    rid = row.get("run_id", 0)
    
    # 紀錄第 0 區塊的點擊率作為基準
    if row.get("block_idx", 0) == 0:
        baseline_click[rid] = row["click_rate"]
        
    # 計算相對點擊率
    base_cr = baseline_click.get(rid, row["click_rate"])
    row["relative_click_rate"] = row["click_rate"] / base_cr if base_cr > 0 else 1.0
    
    # 累積 Z/X 錯誤率
    row["cumul_z_qber"] = cumulative_z_err[rid] / cumulative_z_total[rid] if cumulative_z_total[rid] > 0 else row.get("z_qber", row["qber"])
    row["cumul_x_qber"] = cumulative_x_err[rid] / cumulative_x_total[rid] if cumulative_x_total[rid] > 0 else row.get("x_qber", row["qber"])
    
    # 更新累積值供下一個 block 使用
    # 假設總數為 10000 光子，依照 QBER 估算錯誤數
    block_photons = 10000 
    cumulative_z_total[rid] += block_photons / 2
    cumulative_x_total[rid] += block_photons / 2
    cumulative_z_err[rid] += (block_photons / 2) * row.get("z_qber", row["qber"])
    cumulative_x_err[rid] += (block_photons / 2) * row.get("x_qber", row["qber"])

# 更新模型要吃的新特徵欄位
EXTENDED_FEATURES = FEATURE_COLUMNS + ["relative_click_rate", "cumul_z_qber", "cumul_x_qber"]

train_rows, test_rows = train_test_split(raw_rows, test_size=0.25, seed=42)   

# ==========================================
# 2. 訓練完整模型（使用擴充特徵）
# ==========================================
model = SimpleDecisionTree(
    max_depth=5,  # 稍微加深以利用新特徵
    min_samples_leaf=5,
    feature_columns=EXTENDED_FEATURES,
)
model.fit(train_rows)

# ==========================================
# 3. 預測與門檻計算
# ==========================================
actual = [row["label"] for row in test_rows]
predicted = [model.predict_one(row) for row in test_rows]

# 找出符合 5% 誤報率的 QBER 門檻
target = 0.05
candidates = sorted({row["qber"] for row in train_rows if row["label"] == "normal"})
best_t=None
for t in candidates:
    if flag_rate_by_class(train_rows, [row["qber"] > t for row in train_rows])["normal"] <= target:
        best_t = t
        break

if best_t is None:
    raise ValueError("找不到符合目標誤報率的門檻")

protocol_t = 0.11
tree_flagged = [p == "intercept_resend" for p in predicted]
qber_flagged = [row["qber"] > best_t for row in test_rows]

methods = [
    ("完整模型 (決策樹+相對特徵)", "—", tree_flagged),
    ("QBER 門檻 (目標 5%)", f"{best_t:.4f}", qber_flagged),
    ("協定門檻", "0.11", [row["qber"] > protocol_t for row in test_rows]),
]

# ==========================================
# 4. 印出第一張表：整體成效與 Z 軸相位型漂移
# ==========================================
print("\n=== 竊聽偵測比較表 (加入相對基準特徵) ===")
labels = sorted({row["label"] for row in test_rows})
non_attack = [label for label in labels if label != "intercept_resend"]

# 將輸出名詞統一替換為教授要求的「Z軸相位型漂移」
display_names = {
    "phase_drift": "Z軸相位型漂移",
    "normal": "正常",
    "aging": "老化"
}

header = ["方法", "門檻", "竊聽檢出率", "非竊聽整體誤報率"] + [f"{display_names.get(label, label)}誤報率" for label in non_attack]
print("| " + " | ".join(header) + " |")
print("|" + "---|" * len(header))

for name, threshold, flagged in methods:
    rate = flag_rate_by_class(test_rows, flagged)
    overall = overall_false_alarm(test_rows, flagged)
    cells = [name, threshold, f"{rate['intercept_resend']:.1%}", f"{overall:.1%}"]
    cells += [f"{rate[label]:.1%}" for label in non_attack]
    print("| " + " | ".join(cells) + " |")

# ==========================================
# 5. 繪製檢出率對攔截比例 f 曲線 (以表格呈現)
# ==========================================
print("\n=== 檢出率對攔截比例 f 曲線資料 ===")
f_stats = defaultdict(lambda: {"total": 0, "detected": 0})

for row, is_flagged in zip(test_rows, tree_flagged):
    if row["label"] == "intercept_resend":
        # 假設資料中 f 值記錄在 intercept_ratio 欄位 (若無則預設為 1.0)
        f_val = row.get("intercept_ratio", 1.0)
        # 取到小數第一位分組
        f_group = round(f_val, 1)
        f_stats[f_group]["total"] += 1
        if is_flagged:
            f_stats[f_group]["detected"] += 1

print("| 攔截比例 (f) | 測試樣本數 | AI 檢出率 |")
print("|---|---|---|")
for f_group in sorted(f_stats.keys()):
    total = f_stats[f_group]["total"]
    detected = f_stats[f_group]["detected"]
    det_rate = detected / total if total > 0 else 0
    print(f"| {f_group:.1f} | {total} | {det_rate:.1%} |")
print()