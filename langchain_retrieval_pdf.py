import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import SimpleSequentialChain, LLMChain, ConversationChain, RetrievalQA
from langchain.globals import set_debug
from langchain.memory import ConversationSummaryMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

# Carregar variáveis de ambiente
load_dotenv()

# Habilitar debug
set_debug(True)

# Configurar o modelo de linguagem
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Carregar documentos PDF
carregadores = [
    PyPDFLoader("GTB_standard_Nov23.pdf"),
    PyPDFLoader("GTB_gold_Nov23.pdf"),
    PyPDFLoader("GTB_platinum_Nov23.pdf")
]

documentos = []
for carregador in carregadores:
    documentos.extend(carregador.load())

# Dividir documentos em partes menores
quebrador = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
textos = quebrador.split_documents(documentos)

# Criar embeddings para os textos
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(textos, embeddings)

# Configurar a cadeia de perguntas e respostas
qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

# Fazer uma pergunta ao modelo
pergunta = "Como devo proceder caso tenha um item comprado roubado?"
resultado = qa_chain.invoke({"query": pergunta})
print(resultado)
