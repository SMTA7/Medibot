from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embedding
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from dotenv import load_dotenv
import os
from src.prompt import *

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
embeddings = download_hugging_face_embedding()

index_name = "medibot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(
    model="E:\Medibot\Model\llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    confiq={'max_new_tokens': 512,
            'temperature': 0.8
            })

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template('index.html')

# Add this API endpoint to process chat requests
@app.route("/get", methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    input=msg
    print(input)
    result = qa.invoke({"query": input})
    print('Response: ', result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)