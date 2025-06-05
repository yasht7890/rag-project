import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# Load API keys from .env
load_dotenv()
client = OpenAI()

st.set_page_config(page_title="📄 PDF Chatbot", layout="centered")
st.title("📚 Ask Your PDF")

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# If file uploaded
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("✅ PDF uploaded. Now indexing...")

    # Step 2: Load and Split PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # Step 3: Embed and Index with Qdrant
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = QdrantVectorStore.from_documents(
        docs,
        embedding=embeddings,
        url="http://localhost:6333",  # ensure Qdrant is running
        collection_name="streamlit_vectors"
    )

    st.success("✅ Indexed! You can now ask questions.")

    # Step 4: Take Query
    query = st.text_input("❓ Ask something from the document")

    if query:
        # Search similar chunks
        results = vectorstore.similarity_search(query)

        context = "\n\n".join([
            f"Page: {r.metadata.get('page_label', '?')}\nContent: {r.page_content}"
            for r in results
        ])

        system_prompt = f"""
        You are a helpful assistant. Use only the context below to answer the question.
        Always refer to the page number if possible.

        Context:
        {context}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # Get Answer
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # or "gpt-3.5-turbo"
            messages=messages
        )

        answer = response.choices[0].message.content

        st.markdown("### 🤖 Answer")
        st.write(answer)
