#!/bin/bash

set -euo pipefail

# 顏色輸出
GREEN='\033[0;32m'
NC='\033[0m' # 無色

log() {
  echo -e "${GREEN}[INFO] $1${NC}"
}

# 安裝系統依賴
log "更新系統並安裝必要套件"
sudo apt update -y
sudo apt install -y build-essential wget curl python3.12-venv

# 建立與啟用虛擬環境
VENV_DIR=".venv"
log "建立與啟用 Python 虛擬環境"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
else
  log "虛擬環境已存在，略過建立"
fi
source "$VENV_DIR/bin/activate"

# 安裝 Python 套件
log "安裝 Python 依賴套件"
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric==2.6.1
pip install ogb==1.3.6

# experiment
pip install neo4j-rust-ext==5.28.1.0 python-dotenv==1.1.0 matplotlib

# 額外（可選）PyG 套件：
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

log "✅ 所有步驟完成"
