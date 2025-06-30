#!/usr/bin/env bash
set -e

echo "🔌 Starting Openfabric event server on port 8888…"
# Run ignite.py in the background
poetry run python ./ignite.py &  
IGNITE_PID=$!

# Give the server a moment to come up
sleep 2

echo "🚀 Launching Streamlit app on port 8501…"
exec poetry run streamlit run ./streamlit_app.py --server.port 8501 --server.headless true
