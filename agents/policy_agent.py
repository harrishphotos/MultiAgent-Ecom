from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from config import LLM_MODEL, EMBEDDING_MODEL, POLICY_FILE_PATH

def create_policy_agent_chain():
    """Create a RAG chain for answering policy-related questions."""

    # 1. initializing the Ollama models
    llm = Ollama(model=LLM_MODEL)
    embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # 2. import the policy document
    with open(POLICY_FILE_PATH, 'r', encoding='utf-8') as f:
        policy_text = f.read() 
    # 3. we need to create a Document and split it into chucks 
    docs = [Document(page_content=policy_text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    split_docs = text_splitter.split_documents(docs)

    # 4. we are converting docs array to arrays of vectors using FAISS to make symatic search
    vector_store = FAISS.from_documents(split_docs, embeddings_model)

    # 5. creating a retriever to make actual search
    retriever = vector_store.as_retriever()

    # 6. creating a RAG prompt 
    rag_prompt = ChatPromptTemplate.from_template(
        """Answer the user's question based ONLY on the following context:

        Context:
        {context}

        Question: {input}
        """
    )

    question_answer_chain = create_stuff_documents_chain(llm, rag_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain
    
