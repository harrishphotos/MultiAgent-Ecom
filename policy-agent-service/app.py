from pathlib import Path
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import EMBEDDING_MODEL, POLICY_FILE_PATH, OLLAMA_BASE_URL

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory where the FAISS vector store will be saved/loaded
VECTOR_STORE_DIR = Path(__file__).resolve().parent / "vector_store"


class PolicyQuery(BaseModel):
    query: str


app = FastAPI()


# --- Initialization Function (No Retry) ---
def initialize_policy_agent():
    """
    Initializes the policy agent once during startup.
    No retry logic.
    """
    try:
        logger.info("Initializing Policy Agent...")

        # 1. Initialize Ollama model
        embeddings_model = OllamaEmbeddings(
            model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL
        )
        logger.info("Ollama model connected.")

        # 2. Load or create vector store
        if (VECTOR_STORE_DIR / "index.faiss").exists():
            logger.info("Loading existing vector store from %s ...", VECTOR_STORE_DIR)
            vector_store = FAISS.load_local(
                str(VECTOR_STORE_DIR),
                embeddings_model,
                allow_dangerous_deserialization=True,
            )
        else:
            logger.info("Creating new vector store (first run)...")

            with open(POLICY_FILE_PATH, "r", encoding="utf-8") as f:
                policy_text = f.read()
            logger.info("Policy document loaded.")

            docs = [Document(page_content=policy_text)]
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=100, chunk_overlap=10
            )
            split_docs = text_splitter.split_documents(docs)
            logger.info("Documents split into chunks.")

            VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
            vector_store = FAISS.from_documents(split_docs, embeddings_model)
            vector_store.save_local(str(VECTOR_STORE_DIR))
            logger.info("Vector store saved to %s", VECTOR_STORE_DIR)

        retriever = vector_store.as_retriever()
        logger.info("Vector store ready.")

        logger.info("✅ Policy agent initialized successfully.")
        return retriever

    except Exception as e:
        logger.critical("❌ Failed to initialize policy agent: %s", e)
        return None


# --- Application Startup Event ---
@app.on_event("startup")
def on_startup():
    app.state.retriever = initialize_policy_agent()


@app.post("/policy_query")
async def policy_query_api(policy_query: PolicyQuery, request: Request):
    if not policy_query.query:
        return {"error": "Query not provided"}

    retriever = request.app.state.retriever
    if retriever is None:
        return {
            "error": "retriever is not available due to an initialization failure. Please check the service logs."
        }

    response = await retriever.ainvoke(policy_query.query)
    context = "\n\n".join([doc.page_content for doc in response])
    return {"policy_context": context}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
