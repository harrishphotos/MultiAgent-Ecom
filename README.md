# MultiAgent E-commerce System

A multi-agent e-commerce customer support system built with FastAPI and LangChain that intelligently routes customer queries to specialized agents.

## Architecture

The system consists of three microservices:

- **Router Agent** (Port 8002) - Main entry point with web UI that routes queries to appropriate agents
- **Order Agent** (Port 8000) - Handles order status lookups  
- **Policy Agent** (Port 8001) - Answers refund policy and general questions using RAG

## Features

- 🤖 Intelligent query routing using LLM-based intent recognition
- 📦 Order status tracking with mock database
- 📋 Policy question answering with RAG retrieval
- 🌐 Clean web interface for customer interactions
- 🐳 Docker-based microservices architecture

## Quick Start

1. **Prerequisites**: Docker, Docker Compose, and Ollama running locally with `llama3:8b` and `mxbai-embed-large` models

2. **Start the system**:
   ```bash
   docker compose up --build
   ```

3. **Access the web interface**: http://localhost:8002

## Usage Examples

Try these queries in the web interface:

- **Order Status**: "What's the status of order ORD123?"
- **Refund Policy**: "What is your refund policy?"  
- **General**: "How long does shipping take?"

## Configuration

Modify `config.py` to change:
- LLM model (`llama3:8b`)
- Embedding model (`mxbai-embed-large`)
- Ollama base URL (`http://host.docker.internal:11434`)

## Tech Stack

- **Backend**: FastAPI, LangChain, FAISS
- **LLM**: Ollama (llama3:8b)
- **Frontend**: Jinja2 templates
- **Infrastructure**: Docker, Docker Compose