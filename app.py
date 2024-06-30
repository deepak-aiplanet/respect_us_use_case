import streamlit as st
import os, json 
from langchain.document_loaders import OnlinePDFLoader
from PyPDF2 import PdfReader
from graphviz import Digraph
from langchain.embeddings import HuggingFaceEmbeddings
import graphviz
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_community.chat_models import AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
import os, tempfile
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
st.set_page_config(page_title="Decision Tree")    

def load_data(file_content):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file_content)
        temp_pdf_path = temp_pdf.name

    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=80)
    text_chunks = text_splitter.split_documents(docs)

    # embeddings = HuggingFaceInferenceAPIEmbeddings(
    #                 api_key = HF_token,
    #                 model_name = "BAAI/bge-base-en-v1.5"
    #                 )
    # vectordb = Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory=persist_directory)
    # vectordb.persist()

    return text_chunks

def load_retrievers(text_chunks):
    bm25_retriever = BM25Retriever.from_documents(text_chunks)
    tfidf_retriever = TFIDFRetriever.from_documents(text_chunks)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever,tfidf_retriever], weights=[0.5,0.5]
    )
    return ensemble_retriever

def load_conv_chain(ensemble_retriever):
    template = """
    <|system|>>
   You are a business analyst with extensive knowledge of legal documents and regulatory documentation.
    Your expertise is in helping people understand and extract key information from such documents. 
    Your task is to extract the rules and exceptions in a way that enables the creation of a decision tree, facilitating integration into the proper flow.
    Legal Document Context: {context}
    Response Structure:
    Create a decision tree in JSON format based on the following structure:

    Write a question and question should be two response like yes or no. if yes it has fallowing answers or other question 
        - If Yes, the result should be: "Not restricted" additional -Council regulations: provide dates and articles if possible.
        - If No, proceed to the next question2.( by giving some link to the next question not direct to next question)

    2. Next question based on the previous question outcome.
        - If Yes, the result should be: "Not restricted" additional -Council regulations: provide dates and articles if possible.
        - If No, proceed to the next question.
    In simple terms - flow chat if conditons.
    [Continue this structure for as many questions as needed, ensuring each question branches into Yes/No answers and provides appropriate results based on the Council regulations.]
    Please continue this format for as many questions as needed, ensuring each question follows the same structure.
    Output is the JSON response follow this pattern: Do not change everytime Json output

   
    Additional Instructions:
    
    Analyze the entire document to identify all relevant rules and exceptions.
    Ensure that the descriptions of rules and exceptions are clear and concise.
    Include relevant dates, jurisdictions, and specific regulations where applicable.
    Structure the questions and answers to facilitate the creation of a logical decision tree or workflow.
    If the regulation mentions specific products, territories, or operations, include them in the appropriate sections.
    Aim to simplify legal language while maintaining accuracy and intent.
    [Provide your answer in JSON form. Reply with only the answer in JSON form and include no other commentary]:
    Provide your answer in JSON form. Reply with only the answer in JSON form and include no other commentary
    Return Valid Json to create Tree. 

     This is JSON output Example, add more questions in this formate only. 
    {{
        "Question1": ,
        "Yes": {{
            "Result": ,
            "Council regulations": 
        }},
        "No": {{
            "Question2": ,
            "Yes": {{
                "Result":,
                "Council regulations": 
            }},
            "No": {{
                "Question3": ,
                "Yes": {{
                    "Result": ,
                    "Council regulations":
                }},
                "No": {{
                    "Result": ,
                    "Council regulations": 
                }}
            }}
        }}
    }}


    CONTEXT: {context}
    </s>
    <|user|>
    {query}
    </s>
    <|assistant|>
    """
    
    BASE_URL = "https://gpt-res.openai.azure.com"
    DEPLOYMENT_NAME = "gpt4-inference"
    API_KEY = 'a20bc67dbd7c47ed8c978bbcfdacf930'
    llm = AzureChatOpenAI(
        openai_api_base=BASE_URL,
        openai_api_version="2023-07-01-preview",
        deployment_name=DEPLOYMENT_NAME,
        openai_api_key=API_KEY,
        model_name = 'gpt-4-32k',
        openai_api_type="azure",
        )

    # vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
    # compressor = CohereRerank(cohere_api_key=st.secrets['COHERE_API_KEY'] )
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor = compressor, 
    #     base_retriever = ensemble_retriever)

    # Define the path to the pre-trained model you want to use
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':'cpu'}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )


    vectorstore = Chroma.from_documents(documents=ensemble_retriever, embedding=embeddings)

    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
            
    chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
    )

    return chain

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def json_to_dot(json_obj, parent=None, graph=None, node_id=0):
    if graph is None:
        graph = graphviz.Digraph()

    if parent is None:
        parent = 'root'
        graph.node(parent, label="root")

    current_node = f'node{node_id}'
    if isinstance(json_obj, dict):
        graph.node(current_node, label="{}")
        graph.edge(parent, current_node)
        for key, value in json_obj.items():
            child_node = f'{current_node}_{key}'
            graph.node(child_node, label=key)
            graph.edge(current_node, child_node)
            json_to_dot(value, child_node, graph, node_id + 1)
    elif isinstance(json_obj, list):
        graph.node(current_node, label="[]")
        graph.edge(parent, current_node)
        for index, item in enumerate(json_obj):
            child_node = f'{current_node}_{index}'
            graph.node(child_node, label=f'[{index}]')
            graph.edge(current_node, child_node)
            json_to_dot(item, child_node, graph, node_id + 1)
    else:
        graph.node(current_node, label=str(json_obj))
        graph.edge(parent, current_node)

    return graph

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(text)

    return text_chunks

def main():

    st.header("Decision Tree from the PDF")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if st.button("Submit & Process"):

        with st.spinner("Processing..."):

            text_chunks = load_data(uploaded_file.read())
            # ensemble_retriever = load_retrievers(text_chunks)
            chain = load_conv_chain(text_chunks)

            decision_tree_json = chain.invoke("Give Decision tree in the document")
            response = json.loads(decision_tree_json)
            graph = json_to_dot(response)
            st.graphviz_chart(graph.source)


if __name__ == '__main__':
    main()
