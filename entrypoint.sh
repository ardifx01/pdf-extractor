#!/bin/sh

MODEL_DIR="/app/app/models"
CACHE_DIR="/root/.cache/docling/models"

# Function to check if a directory has files
dir_has_files() {
    [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]
}

# Check if models exist in either CACHE_DIR or MODEL_DIR
if dir_has_files "$CACHE_DIR" || dir_has_files "$MODEL_DIR"; then
    echo "✅ Models already cached in $CACHE_DIR or exist in $MODEL_DIR"
else
    echo "🔽 Downloading models to cache..."
    docling-tools models download layout tableformer code_formula easyocr
fi

# Ensure MODEL_DIR has the models
if ! dir_has_files "$MODEL_DIR"; then
    echo "📦 Copying models from cache to volume $MODEL_DIR ..."
    mkdir -p "$MODEL_DIR"
    cp -r "$CACHE_DIR"/* "$MODEL_DIR"/ 2>/dev/null || true
else
    echo "✅ Model already exists in $MODEL_DIR. Skipping copy."
fi

exec "$@"
