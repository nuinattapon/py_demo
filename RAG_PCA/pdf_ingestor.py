import os
import shutil
from typing import List
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
from dotenv import load_dotenv
import glob

# Load environment variables
load_dotenv()

def extract_text_from_pdf_with_page_numbers(pdf_path: str) -> list[dict[int,str]]:
    """
    Extract text from a PDF file with page numbers.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of tuples containing (page_number, text)
    """
    pages_text = []
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                if text.strip():  # Only add pages with text
                    pages_text.append((page_num, text))
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return []
    return pages_text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of specified size with overlap.
    
    Args:
        text (str): Text to chunk
        chunk_size (int): Size of each chunk in words
        overlap (int): Number of words to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    # Split text into words
    words = text.split()
    chunks = []
    
    # If text is empty or chunk_size is 0, return the whole text as one chunk
    if not words or chunk_size <= 0:
        return [text]
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        # Join words to form chunk
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        # Make sure we don't go beyond the text
        if start >= len(words):
            break
    return chunks

def create_bge_embedding_function(model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Create an embedding function using BGE model.
    
    Args:
        model_name (str): Name of the BGE model to use
        
    Returns:
        embedding_functions.EmbeddingFunction: Embedding function for ChromaDB
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )

def ingest_all_pdfs_to_chromadb(pdf_directory: str = "pca_documents", collection_name: str = "pca_documents"):
    """
    Ingest all PDF documents from a directory into ChromaDB with BGE embeddings.
    
    Args:
        pdf_directory (str): Path to the directory containing PDF files
        collection_name (str): Name of the ChromaDB collection
    """
    # Initialize ChromaDB client with persistent storage
    db_path = "./chroma_db"
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        print(f"Removing existing database at {db_path}")
        shutil.rmtree(db_path)
    
    client = chromadb.PersistentClient(path=db_path)
    
    # Create embedding function
    embedding_function = create_bge_embedding_function("BAAI/bge-base-en-v1.5")
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Get all PDF files in the directory
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {pdf_directory}")
    
    total_chunks = 0
    chunk_id = 0
    
    # Process each PDF file
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\nProcessing {filename}...")
        
        # Extract text from PDF with page numbers
        pages_text = extract_text_from_pdf_with_page_numbers(pdf_path)
        print(f"Extracted text from {len(pages_text)} pages")
        
        if not pages_text:
            print(f"Failed to extract text from {filename}")
            continue
        
        # Process each page
        for page_num, page_text in pages_text:
            # Chunk text
            chunks = chunk_text(page_text, chunk_size=500, overlap=50)
            print(f"  Page {page_num}: Created {len(chunks)} chunks")
            
            # Create IDs and metadata for chunks
            ids = [f"chunk_{chunk_id + i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source": filename,
                    "page_number": page_num,
                    "chunk_index": i
                } 
                for i in range(len(chunks))
            ]
            
            # Add documents to collection
            if chunks:  # Only add if there are chunks
                collection.add(
                    documents=chunks,
                    ids=ids,
                    metadatas=metadatas
                )
                chunk_id += len(chunks)
                total_chunks += len(chunks)
    
    print(f"\nSuccessfully ingested all PDFs into ChromaDB collection '{collection_name}'")
    print(f"Collection now contains {collection.count()} documents")

def query_collection(query_text: str, collection_name: str = "pca_documents", n_results: int = 5):
    """
    Query the ChromaDB collection.
    
    Args:
        query_text (str): Text to search for
        collection_name (str): Name of the ChromaDB collection
        n_results (int): Number of results to return
    """
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Get collection
    collection = client.get_collection(name=collection_name)
    
    # Query collection
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    print(f"\nQuery: {query_text}")
    print(f"Top {n_results} results:")
    if results and 'documents' in results and results['documents']:
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            print(f"\n{i+1}. Distance: {distance:.4f}")
            print(f"   Metadata: {metadata}")
            print(f"   Content: {doc[:200]}...")
    else:
        print("No results found")
    
    return results

if __name__ == "__main__":
    # Ingest all PDFs into ChromaDB
    ingest_all_pdfs_to_chromadb()
    
    # Example query
    query_collection("What are the main principles of PCA?")
