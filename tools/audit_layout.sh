#!/bin/bash
# 检查一个项目是否符合 docs/REPO_LAYOUT.md。无输出即合规。
# 用法: bash tools/audit_layout.sh 2Dphi4
P="${1:?用法: audit_layout.sh <项目目录>}"
P="${P%/}"

# runs/ trainingdata/ lightning_logs/ references/ 非本人维护，规范不管其内部
PRUNE=( -path "*/runs/*" -o -path "*/trainingdata/*" -o -path "*/lightning_logs/*" -o -path "*/references/*" -o -path "*/.git/*" )

echo "== 规则 A: 保留名出现在非法位置 =="
find "$P" \( "${PRUNE[@]}" \) -prune -o -type d \
  \( -name results -o -name dat -o -name data -o -name figures -o -name logs \) -print \
  | grep -vE "^$P/(results|logs)$"

echo "== 规则 B: 人写的目录超过 3 层 =="
find "$P" \( "${PRUNE[@]}" -o -path "$P/results/*" \) -prune -o -type d -print \
  | awk -F/ 'NF>3'

echo "== 规则 C: run 目录内的非法子目录 =="
find "$P/runs" -mindepth 2 -maxdepth 2 -type d 2>/dev/null \
  | grep -vE '/(models|samples|logs|lightning_logs)$'

# 只查空格。大写不查: L128、Lmulti、ESS、HMC 是物理符号和缩写，本就该保留
echo "== 目录名含空格 =="
find "$P" \( "${PRUNE[@]}" \) -prune -o -type d -print | grep ' '

echo "== 代码目录里混入产物 =="
find "$P"/{analysis,train,sample,solver,tests} -type f \
  \( -name '*.npz' -o -name '*.npy' -o -name '*.pdf' -o -name '*.png' -o -name '*.dat' -o -name '*.csv' \) 2>/dev/null
