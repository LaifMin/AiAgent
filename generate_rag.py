from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore


#Carico il documento
loader = PyPDFLoader("vs\carroll_alice_nel_etc_loescher.pdf")
docs = loader.load()
print("Documenti caricati", len(docs))

#split in parti piu piccole
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 500,
    length_function = len,
    is_separator_regex=False
)


chunks = text_splitter.split_documents(docs)
print("chunk",len(chunks))


embeddings = OllamaEmbeddings(model = "embeddinggemma:300m")
vs = InMemoryVectorStore.from_documents(chunks, embeddings)
print("vector creato")

vs.dump("./vs/alice.db")
print("vec store salvato")