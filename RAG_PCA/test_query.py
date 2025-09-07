import chromadb
from dotenv import load_dotenv
from llm_utils import generate_with_multiple_input

def test_query():
    """Test querying the ChromaDB collection and using LLM to generate response."""
    # Load environment variables
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Get collection
    collection = client.get_collection(name="pca_documents")
    
    print(f"Collection '{collection.name}' contains {collection.count()} documents")
    
    # Query collection
    query_text = "How many nodes we can have in OKE cluster?"
    results = collection.query(
        query_texts=[query_text],
        n_results=5
    )
    
    print(f"\nQuery: {query_text}")
    print(f"Top 5 results:")
    
    # Prepare context from retrieved documents
    context = ""
    if results and 'documents' in results and 'metadatas' in results and 'distances' in results:
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        distances = results.get('distances', [])
        
        if documents and metadatas and distances and len(documents) > 0:
            for i, (doc, metadata, distance) in enumerate(zip(
                documents[0], 
                metadatas[0], 
                distances[0]
            )):
                print(f"\n{i+1}. Distance: {distance:.4f}")
                source = metadata.get('source', 'N/A') if metadata else 'N/A'
                page_number = metadata.get('page_number', 'N/A') if metadata else 'N/A'
                print(f"   Source: {source}")
                print(f"   Page: {page_number}")
                content = doc[:500] if doc else "No content"
                print(f"   Content: {content}")
                # Add to context for LLM
                context += f"Document {i+1} (Source: {source}, Page: {page_number}):\n{doc}\n\n"
    else:
        print("Unexpected results format")
        print(f"Available keys: {list(results.keys()) if results else 'No results'}")
        return results
    
    # Use LLM to generate response based on retrieved context
    print("\n" + "="*50)
    print("Generating response with LLM...")
    print("="*50)
    
    # Create prompt for LLM
    prompt = f"""
    Based on the following documents, please answer the question: "{query_text}"
    
    Documents:
    "{context}"
    
    Please provide a comprehensive answer based only on the information in the documents above.
    If the information is not available in the documents, please state that.
    """
    
    # Prepare messages for LLM
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents."},
        {"role": "user", "content": prompt}
    ]
    
    # Call LLM using TogetherAI
    try:
        llm_response = generate_with_multiple_input(
            messages=messages,
            model="openai/gpt-oss-20b",
            temperature=0,
            max_tokens=1000
        )
        
        print("\nLLM Response:")
        print("-" * 30)
        print(llm_response["content"])
        
    except Exception as e:
        print(f"Error calling LLM: {e}")
        print("Returning raw results without LLM processing.")
    
    return results

if __name__ == "__main__":
    load_dotenv()
    test_query()
