# === 設定變數 ===
PLUGIN_NAME := neotorch
VERSION := 1.0.0-SNAPSHOT
JAR_NAME := $(PLUGIN_NAME)-$(VERSION).jar
NEO4J_HOME := /var/lib/neo4j
PLUGIN_DIR := $(NEO4J_HOME)/plugins
JAR_PATH := target/$(JAR_NAME)
DEPLOY_PATH := $(PLUGIN_DIR)/$(JAR_NAME)
BACKUP_PATH := $(DEPLOY_PATH).old

# === 預設目標 ===
.PHONY: all build deploy clean restart help

all: build deploy restart

# === 編譯並打包 JAR ===
build:
	@echo "🔨 編譯並打包 $(JAR_NAME)..."
	@mvn clean package

# === 備份舊版 JAR 並部署新版 JAR ===
# === 複製 model builder 到 plugins 目錄 ===
deploy: $(JAR_PATH)
	@echo "🚚 部署 $(JAR_NAME) 到 $(PLUGIN_DIR)..."
	@if [ -f $(DEPLOY_PATH) ]; then \
		echo "📦 備份舊版 JAR 為 $(BACKUP_PATH)"; \
		sudo -u neo4j mv $(DEPLOY_PATH) $(BACKUP_PATH); \
	fi
	@sudo -u neo4j cp $(JAR_PATH) $(DEPLOY_PATH)

# === 清除編譯產物 ===
clean:
	@echo "🧹 清除編譯產物與部署檔案..."
	@mvn clean
	@sudo -u neo4j rm -f $(DEPLOY_PATH) $(BACKUP_PATH)

# === 重啟 Neo4j 服務（請根據實際環境調整）===
restart:
	@echo "🔁 重啟 Neo4j 服務..."
	@sudo systemctl restart neo4j || echo "⚠️ 請手動重啟 Neo4j 服務"
	@sudo journalctl -u neo4j -f -n 10

# === 顯示可用目標 ===
help:
	@echo "📘 可用目標："
	@echo "  make build    - 編譯並打包 JAR"
	@echo "  make deploy   - 備份舊版並部署新版 JAR"
	@echo "  make clean    - 清除編譯產物和部署的 JAR"
	@echo "  make restart  - 重啟 Neo4j 服務（需自行實作）"
	@echo "  make help     - 顯示此說明"

