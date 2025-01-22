from src.helper import load_pdf_file,text_split,download_huggingface_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()


extracted_data=load_pdf_file(data='Data/')
text_chunk=text_split(extracted_data)
embeddings=download_huggingface_embeddings()
api_key = os.getenv("PINECONE_API_KEY")

api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENV")

index_name="rag-pinecon"

pc=Pinecone(api_key=api_key)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws',region=env)
    )


docsearch=PineconeVectorStore.from_documents(
    documents=text_chunk,
    index_name=index_name,
    embedding=embeddings
)
