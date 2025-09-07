import chromadb
import os

def verify_ingestion():
    """Verify that the PDF was successfully ingested into ChromaDB."""
    # Check if chroma_db directory exists
    if not os.path.exists("./chroma_db"):
        print("Error: chroma_db directory not found")
        return False
    
    print("✓ chroma_db directory exists")
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # List all collections
    collections = client.list_collections()
    print(f"Available collections: {[c.name for c in collections]}")
    
    # Check if our collection exists
    try:
        collection = client.get_collection(name="pca_documents")
        print(f"✓ Collection 'pca_documents' found")
        
        # Get document count
        count = collection.count()
        print(f"✓ Collection contains {count} documents")
        
        if count > 0:
            print("✓ PDF ingestion was successful!")
            return True
        else:
            print("⚠ Collection exists but contains no documents")
            return False
            
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return False

if __name__ == "__main__":
    success = verify_ingestion()
    if success:
        print("\n🎉 PDF ingestion verification completed successfully!")
    else:
        print("\n❌ PDF ingestion verification failed!")
