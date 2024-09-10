import os.path
import time
import pandas as pd
import dotenv
import argparse

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnableParallel
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import CacheBackedEmbeddings

from utils import clean_table_with_llm


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rag_cleaning_chain(table: pd.DataFrame, wiki_docs, true_errors: pd.Series, model: str, chunk_size: int,
                       chunk_overlap: int, topk: int, emb_path: str, return_docs: bool = False, temperature: int = 0):
    # Load the model
    dotenv.load_dotenv()
    llm = ChatOpenAI(model=model, temperature=temperature)

    # Create document chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(wiki_docs)

    # Embed the Wiki chunks and store them in the vector store
    embeddings = OpenAIEmbeddings()
    namespace = f'wiki{len(wiki_docs)}_{embeddings.model}'
    emb_store = LocalFileStore(os.path.join(emb_path, namespace))
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings, emb_store, namespace=namespace
    )

    start_emb = time.time()
    vectorstore = FAISS.from_documents(documents=chunks, embedding=cached_embedder)
    # sim_docs = vectorstore.similarity_search(question)
    print(f"Emb time: {time.time() - start_emb}")

    # Create the retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": topk})
    # retrieved_docs = retriever.invoke(question)
    # retrieved_docs[0].page_content

    # Create the prompt
    prompt_template = """Given some information about the following famous person, find an error in the reported
     information (if any) and return the correct record. 
     Use the following format for the answer:

     Error: <reason of the error or None if no error occurred> 
     New record: <the content of the new record in JSON> 

     Stick strictly to the specified output format and don't add extra text because a regular expression will be used to
     parse the output.
     To support your decision use the following pieces of context:
     {context}
     This is the information to check:
     {record}
     """
    prompt = PromptTemplate.from_template(prompt_template)

    # Create the chain
    if not return_docs:
        rag_chain = (
                {"context": retriever | format_docs, "record": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

    else:
        rag_chain_from_docs = (
                RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                | prompt
                | llm
                | StrOutputParser()
        )

        rag_chain = RunnableParallel(
            {"context": retriever, "record": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

    # Ask the LLM to find errors
    out_table, out_errors, out_contexts = clean_table_with_llm(
        table=table, llm=rag_chain, true_errors=true_errors, return_docs=return_docs
    )

    return out_table, out_errors, out_contexts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo-0125",
                        help='The name of the model')
    parser.add_argument('--data', type=str, default=os.path.join('data', 'people.csv'),
                        help='The path to the data to clean')
    parser.add_argument('--errors', type=str, default=os.path.join('data', 'errors.csv'),
                        help='The path to the errors')
    parser.add_argument('--wiki_summaries', type=str, default=os.path.join('data', 'wiki_summaries.csv'),
                        help='The path to the Wikipedia summaries')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Size of each document chunk')
    parser.add_argument('--chunk_overlap', type=int, default=200,
                        help='Overlap between document chunks')
    parser.add_argument('--topk', type=int, default=3,
                        help='The number of documents to retrieve for each question')
    parser.add_argument('--return_docs', action='store_true',
                        help='Flag that enables the insertion of retrieved documents in the answer')
    parser.add_argument('--emb_path', type=str, default='./embeddings',
                        help='Path where to save the embeddings')
    args = parser.parse_args()

    table = pd.read_csv(args.data)
    table.drop(['Id'], axis=1, inplace=True)
    table = table.sample(n=len(table), replace=False)
    true_errors = pd.read_csv(args.errors)['Error']

    wiki_loader = CSVLoader(file_path=args.wiki_summaries, encoding='utf-8')
    wiki_docs = wiki_loader.load()

    rag_cleaning_chain(
        table=table, wiki_docs=wiki_docs, true_errors=true_errors,
        model=args.model, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, topk=args.topk,
        emb_path=args.emb_path, return_docs=args.return_docs
    )


if __name__ == '__main__':
    main()
