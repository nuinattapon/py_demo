import chromadb

def test_metadata():
    """Test that metadata (filename and page number) is correctly stored."""
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Get collection
    collection = client.get_collection(name="pca_documents")
    
    print(f"Collection '{collection.name}' contains {collection.count()} documents")
    
    # Get a sample of documents to check metadata
    results = collection.peek(limit=5)
    
    print("\nSample documents with metadata:")
    if results and 'documents' in results and 'metadatas' in results:
        # Check if IDs are available
        ids = results.get('ids', [])
        for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            doc_id = ids[i] if ids and i < len(ids) else f"doc_{i+1}"
            print(f"\n{i+1}. ID: {doc_id}")
            print(f"   Source: {metadata.get('source', 'N/A')}")
            print(f"   Page: {metadata.get('page_number', 'N/A')}")
            print(f"   Chunk: {metadata.get('chunk_index', 'N/A')}")
            print(f"   Content: {doc[:150]}...")
    else:
        print("Unexpected results format")
        print(f"Available keys: {list(results.keys()) if results else 'No results'}")
    
    # Test a query to see metadata in results
    print("\n\nTesting query with metadata:")
    query_results = collection.query(
        query_texts=["What are the main principles of PCA?"],
        n_results=3
    )
    
    if query_results and 'documents' in query_results and 'metadatas' in query_results:
        # Check if IDs are available in query results
        ids = query_results.get('ids', [])
        for i, (doc, metadata, distance) in enumerate(zip(
            query_results['documents'][0], 
            query_results['metadatas'][0], 
            query_results['distances'][0]
        )):
            doc_id = ids[0][i] if ids and len(ids) > 0 and i < len(ids[0]) else f"query_doc_{i+1}"
            print(f"\n{i+1}. ID: {doc_id}")
            print(f"   Distance: {distance:.4f}")
            print(f"   Source: {metadata.get('source', 'N/A')}")
            print(f"   Page: {metadata.get('page_number', 'N/A')}")
            print(f"   Chunk: {metadata.get('chunk_index', 'N/A')}")
            print(f"   Content: {doc[:150]}...")
    else:
        print("No query results found")

if __name__ == "__main__":
    test_metadata()
