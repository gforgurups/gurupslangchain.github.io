from langchain.document_loaders.sitemap import SitemapLoader
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from pypdf import PdfReader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from langchain.schema import Document
import re

def get_website_data(url):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loader = WebBaseLoader(url)
    loader.requests_kwargs = {'verify':False}
    docs = loader.load()
    return docs

def read_pdf_data_as_doc(pdf):
    loader = PyPDFLoader(pdf)
    return loader.load()

def read_pdf_data_as_text(pdf):
    loader = PdfReader(pdf)
    text=""
    for page in loader.pages:
        text+= page.extract_text()
    return text

def create_docs_from_pdffiles(pdfs, unique_id):
    docs=[]
    for file in pdfs:
        text = read_pdf_data_as_text(file)
        docs.append(Document(
            page_content=text, 
            metadata={"name":file.name, 
                      #"id":file.id,
                      "type":file.type,
                      "size":file.size,
                      "unique_id":unique_id
                        }
        ))
    return docs

def split_data(site_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,length_function=len)
    text_chunks = text_splitter.split_text(site_data)
    doc_chunks = text_splitter.create_documents(text_chunks)
    return doc_chunks

def get_embeddings():
    return SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

def create_embeddings(df, embeddings):
    df[2] = df[0].apply(lambda x: embeddings.embed_query(x))
    return df

def push_to_pinecone(env,index,embeddings,chunks_data):

    pinecone.init(
    api_key='',
    environment="gcp-starter"
    )
    Pinecone.from_documents(documents= chunks_data,embedding=embeddings,index_name=index)

def pull_from_pinecone(env,index,embeddings):
    pinecone.init(
    api_key='',
    environment=env
    )
    vector_store = Pinecone.from_existing_index(index_name=index,embedding=embeddings)
    return vector_store

def get_similar_docs(env,index,embeddings, prompt,document_count,unique_id):
    vector_store = pull_from_pinecone(env,index,embeddings)
    unique_id = unique_id.strip()

    return vector_store.similarity_search_with_score(prompt, document_count,{"unique_id":"361810a5c27d42c8a75a526df04cc412"})


def get_llm_final_answer(docs,prompt):
    llm = OpenAI(openai_api_key="")
    chain = load_qa_chain(llm=llm,chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents = docs, question=prompt)
    return response

def get_llm_summary(doc):
    llm = OpenAI()
    chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
    summary = chain.run([doc])
    return summary

def read_data(data):
    df = pd.read_csv(data,delimiter=",",header=None)
    return df

def split_train_test__data(df):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(list(df[2]),list(df[1]),test_size=0.25,random_state=0)
    return X_train, X_test, y_train, y_test

def get_score(svm_classifier, test_data, test_labels):
    score = svm_classifier.score(test_data,test_labels)
    return score

def predict(data):
    model = joblib.load('modelsvm.pk1')
    result = model.predict([data])
    return result[0]


#Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text



#Function to extract data from text
def extracted_data(pages_data):

    template = """Extract all the following values : invoice no., Description, Quantity, date, 
        Unit price , Amount, Total, email, phone number and address from this data: {pages}

        Expected output: remove any dollar symbols {{'Invoice no.': '1001329','Description': 'Office Chair','Quantity': '2','Date': '5/4/2023','Unit price': '1100.00','Amount': '2200.00','Total': '2200.00','Email': 'Santoshvarma0988@gmail.com','Phone number': '9999999999','Address': 'Mumbai, India'}}
        """
    prompt_template = PromptTemplate(input_variables=["pages"], template=template)

    llm = OpenAI(openai_api_key='', temperature=.7)
    full_response=llm(prompt_template.format(pages=pages_data))
    

    #The below code will be used when we want to use LLAMA 2 model,  we will use Replicate for hosting our model...
    
    #output = replicate.run('replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1', 
                           #input={"prompt":prompt_template.format(pages=pages_data) ,
                                  #"temperature":0.1, "top_p":0.9, "max_length":512, "repetition_penalty":1})
    
    #full_response = ''
    #for item in output:
        #full_response += item
    

    #print(full_response)
    return full_response


# iterate over files in
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list):
    
    df = pd.DataFrame({'Invoice no.': pd.Series(dtype='str'),
                   'Description': pd.Series(dtype='str'),
                   'Quantity': pd.Series(dtype='str'),
                   'Date': pd.Series(dtype='str'),
	                'Unit price': pd.Series(dtype='str'),
                   'Amount': pd.Series(dtype='int'),
                   'Total': pd.Series(dtype='str'),
                   'Email': pd.Series(dtype='str'),
	                'Phone number': pd.Series(dtype='str'),
                   'Address': pd.Series(dtype='str')
                    })

    for filename in user_pdf_list:
        

        raw_data=get_pdf_text(filename)
        llm_extracted_data=extracted_data(raw_data)
        pattern = r'{(.+)}'
        match = re.search(pattern, llm_extracted_data, re.DOTALL)

        if match:
            extracted_text = match.group(1)
            # Converting the extracted text to a dictionary
            data_dict = eval('{' + extracted_text + '}')
            print(data_dict)
        else:
            print("No match found.")


        df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
  
    return df
