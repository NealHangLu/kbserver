import os
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
INDEX_DIR = os.environ.get("INDEX_DIR", "./faissdb")  # Default value if not set

DASHSCOPE_API_KEY = "your keys"
if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY environment variable not set.")

from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from langchain_community.embeddings import DashScopeEmbeddings

try:
    qwen_embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=DASHSCOPE_API_KEY)
except Exception as e:
    print(f"Error initializing DashScope embeddings: {e}")
    exit(1)

vectorstore_disk = None

def initialize_index():
    global vectorstore_disk
    index_dir = "./faissdb"
    try:
        if os.path.exists(index_dir):
            vectorstore_disk = FAISS.load_local(index_dir, qwen_embeddings, allow_dangerous_deserialization=True)
        else:
            print("Index directory not found. Creating a new index requires populating it first.")
            vectorstore_disk = None # Indicate that the index is not ready
    except Exception as e:
        print(f"Error initializing index: {e}")
        vectorstore_disk = None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) if docs else ""

@app.route("/query", methods=["GET"])
def query_index():
    global vectorstore_disk
    if vectorstore_disk is None:
        return jsonify({"error": "Index not initialized"}), 500

    try:
        retriever = vectorstore_disk.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 2, "score_threshold": 0.1},
        )
        query_text = request.args.get("text", None)
        if query_text is None:
            return (
                jsonify(
                    {
                        "error": "No text found, please include a ?text=blah parameter in the URL"
                    }
                ),
                400,
            )
        context = format_docs(retriever.get_relevant_documents(query_text))
        print(retriever.get_relevant_documents(query_text))
        response = jsonify({"result": context})
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response, 200
    except Exception as e:
        return jsonify({"error": f"Error during query: {e}"}), 500

if __name__ == "__main__":
    initialize_index()  # Initialize the index before starting the app
    app.run(host="0.0.0.0", port=5601, debug=False)  # Set debug=False in production
