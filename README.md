---
title: OCR & RAG Agent
emoji: 🛢️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8501
---
# Agent
use model : gemma4:26b

## Deploy on Render
This project is now ready for Render deployment using the included `Dockerfile`.

### How to deploy
1. Create a new Render Web Service.
2. Choose `Docker` as the environment.
3. Set the Dockerfile path to `Dockerfile`.
4. Use port `8501`, or let Render set the `PORT` environment variable.
5. Deploy.

### Notes
- The Dockerfile installs the Ollama CLI.
- Streamlit is started with `PORT` support so Render can route traffic correctly.
