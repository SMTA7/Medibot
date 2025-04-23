from src.helper import load_pdf_file , text_split , download_hugging_face_embedding
from langchain.vectorstores import Pinecone
import pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY  = os.environ.get('PINECONE_API_KEY')

extracted_data=load_pdf_file(data='E:\Medibot\Data')

text_chunks=text_split(extracted_data)

embeddings=download_hugging_face_embedding()

index_name = "medibot"


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)

