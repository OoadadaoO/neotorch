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

# 安裝 CUDA 12.4
CUDA_RUNFILE="cuda_12.4.0_550.54.14_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/${CUDA_RUNFILE}"

log "下載 CUDA 12.4 安裝檔案"
if [ ! -f "$CUDA_RUNFILE" ]; then
  wget "$CUDA_URL"
fi
if [ ! -d "/usr/local/cuda-12.4/bin" ]; then
  log "安裝 CUDA 12.4"
  sudo sh "$CUDA_RUNFILE" --silent
else
  log "CUDA 12.4 已安裝，略過安裝"
fi

# 設定 CUDA 環境變數（如未設定）
CUDA_EXPORTS="# CUDA 12.4
export PATH=/usr/local/cuda-12.4/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:\$LD_LIBRARY_PATH"
if ! grep -q "/usr/local/cuda-12.4/bin" ~/.bashrc; then
  echo "$CUDA_EXPORTS" >> ~/.bashrc
  log "已將 CUDA 環境變數新增至 ~/.bashrc"
else
  log "CUDA 環境變數已存在，略過新增"
fi

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
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric==2.6.1
pip install ogb==1.3.6
# pip install neo4j-rust-ext==5.28.1.0

# 額外（可選）PyG 套件：
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

log "✅ 所有步驟完成"
