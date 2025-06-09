#!/bin/bash

set -euo pipefail

# 顏色輸出
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # 無色

log() {
  echo -e "${GREEN}[INFO] $1${NC}"
}
error() {
  echo -e "${RED}[ERROR] $1${NC}"
  exit 1
}

# 確保以 root 執行
if [[ "$EUID" -ne 0 ]]; then
  error "請以 root 執行本腳本 (sudo bash install.sh)"
fi

# 預設 NEO4J_HOME（如未設定）
NEO4J_HOME=$(sudo -u neo4j bash -c 'echo $NEO4J_HOME')
if [[ -z "$NEO4J_HOME" ]]; then
  NEO4J_HOME="/var/lib/neo4j"
fi

PLUGIN_DIR="$NEO4J_HOME/plugins/neotorch"
VENV_DIR="$PLUGIN_DIR/.venv"

# 建立目錄
log "建立插件工作目錄：$PLUGIN_DIR"
sudo -u neo4j mkdir -p "$PLUGIN_DIR"

# 安裝系統依賴
log "安裝必要系統套件"
apt update -y
apt install -y build-essential wget curl python3.12-venv

# CUDA 安裝
CUDA_VER="12.4"
CUDA_RUNFILE="cuda_${CUDA_VER}.0_550.54.14_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_VER}.0/local_installers/${CUDA_RUNFILE}"
CUDA_PATH="/usr/local/cuda-${CUDA_VER}"

if [ ! -d "$CUDA_PATH/bin" ]; then
  log "下載並安裝 CUDA ${CUDA_VER}"
  wget -nc -q --show-progress -O "$CUDA_RUNFILE" "$CUDA_URL"
  sh "$CUDA_RUNFILE" --silent
else
  log "CUDA ${CUDA_VER} 已安裝，略過"
fi

# DJL PyTorch native jar 參數
DJL_PYTORCH_NATIVE="pytorch-native-cu124"
DJL_PYTORCH_NATIVE_VERSION="2.5.1"
DJL_JAR_NAME="${DJL_PYTORCH_NATIVE}-${DJL_PYTORCH_NATIVE_VERSION}-linux-x86_64.jar"
DJL_JAR_URL="https://repo1.maven.org/maven2/ai/djl/pytorch/${DJL_PYTORCH_NATIVE}/${DJL_PYTORCH_NATIVE_VERSION}/${DJL_JAR_NAME}"
DJL_JAR_PATH="${NEO4J_HOME}/plugins/${DJL_JAR_NAME}"

# 如需強制重新下載，將此變數設為 true
FORCE_DOWNLOAD=false

# 下載 DJL jar（如必要）
if [[ "$FORCE_DOWNLOAD" == true || ! -f "$DJL_JAR_PATH" ]]; then
  log "下載 DJL PyTorch native jar：$DJL_JAR_NAME"
  if ! wget -nc -q --show-progress -O "$DJL_JAR_PATH" "$DJL_JAR_URL"; then
    error "無法下載 DJL jar 檔案，請檢查網路或 URL 是否正確"
  fi
else
  log "DJL PyTorch native jar 已存在，略過下載"
fi

# 建立安裝腳本給 neo4j 使用者
INSTALL_SCRIPT="$PLUGIN_DIR/setup_env.sh"

log "產生安裝腳本於 $INSTALL_SCRIPT"
cat <<EOF | sudo -u neo4j tee "$INSTALL_SCRIPT" > /dev/null
#!/bin/bash
set -e

cd "$PLUGIN_DIR"

# 建立虛擬環境
if [ ! -d "$VENV_DIR" ]; then
  python3.12 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# 安裝 Python 套件
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric==2.6.1
EOF

chmod +x "$INSTALL_SCRIPT"

log "以 neo4j 使用者身份執行環境安裝"
sudo -u neo4j bash "$INSTALL_SCRIPT"

# 自動修改 systemd 設定
SYSTEMD_FILE="/usr/lib/systemd/system/neo4j.service"

log "修改 Neo4j systemd service 檔案：$SYSTEMD_FILE"
if grep -q "Environment=.*cuda" "$SYSTEMD_FILE"; then
  log "已存在 CUDA 環境變數設定，略過"
else
  sed -i.bak "/^\[Service\]/a Environment=\"PATH=${CUDA_PATH}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"\nEnvironment=\"LD_LIBRARY_PATH=${CUDA_PATH}/lib64\"" "$SYSTEMD_FILE"
  log "已新增 CUDA 環境變數到 neo4j.service"
fi

# 重載並重啟 Neo4j
log "重新加載 systemd 並重啟 neo4j"
systemctl daemon-reexec
systemctl daemon-reload
systemctl restart neo4j

log "✅ 安裝完成並已重啟 Neo4j"
