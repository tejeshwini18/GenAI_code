from sentence_transformers import SentenceTransformer, util
import time
import hashlib
import torch
import numpy as np
from typing import List, Dict


class SemanticSearchEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the search engine with a specific embedding model.

        Args:
            model_name: The name of the SentenceTransformer model to use
        """
        # Initialize the embedding model
        self.model = SentenceTransformer(model_name)
        # Create empty data structures for document storage
        self.documents = []  # List of document dictionaries
        self.document_embeddings = []  # List of embedding vectors
        # Initialize a cache for query embeddings
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def add_documents(self, documents: List[Dict[str, str]], batch_size: int = 32) -> None:
        """
        Process and add documents to the search engine.

        Args:
            documents: List of document dictionaries with 'id' and 'content' keys
            batch_size: Batch size for efficient embedding generation
        """
        # Process documents in batches of the specified size
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            # Generate embeddings for each batch
            batch_texts = [doc['content'] for doc in batch]
            batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
            # Store documents with their embeddings in your data structure
            self.documents.extend(batch)
            # Convert to list of numpy arrays for storage
            self.document_embeddings.extend([emb for emb in batch_embeddings])

    def _get_embedding(self, text: str, return_tensor: bool = False):
        """
        Get embedding for a text, using cache if available.

        Args:
            text: The text to embed
            return_tensor: If True, return as tensor; if False, return as numpy array

        Returns:
            The embedding vector for the text
        """
        # Create a hash of the input text to use as a cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Check if the embedding exists in the cache
        if text_hash in self.embedding_cache:
            # If it does, increment cache hit counter and return cached embedding
            self.cache_hits += 1
            cached_embedding = self.embedding_cache[text_hash]
            if return_tensor:
                return torch.tensor(cached_embedding)
            return np.array(cached_embedding)
        else:
            # If not, generate the embedding, cache it, increment cache miss counter
            embedding = self.model.encode(text, convert_to_numpy=True)
            self.embedding_cache[text_hash] = embedding.tolist()
            self.cache_misses += 1
            # Return the embedding
            if return_tensor:
                return torch.tensor(embedding)
            return embedding

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Search for documents most similar to the query.

        Args:
            query: The search query text
            top_k: Number of top results to return

        Returns:
            List of top_k documents with their similarity scores
        """
        # Get embedding for the query (using cache if possible)
        query_embedding = self._get_embedding(query, return_tensor=True)
        query_embedding = query_embedding.unsqueeze(0)  # Add batch dimension
        
        # Calculate similarity between query and all documents
        if len(self.document_embeddings) == 0:
            return []
        
        # Convert document embeddings to tensor
        # document_embeddings is a list of numpy arrays
        doc_embeddings_array = np.array(self.document_embeddings)
        doc_embeddings_tensor = torch.tensor(doc_embeddings_array)
        
        # Calculate cosine similarity using util.cos_sim
        # util.cos_sim handles normalization internally
        similarities = util.cos_sim(query_embedding, doc_embeddings_tensor)[0]
        
        # Sort the results by similarity score in descending order
        # Get indices sorted by similarity (descending)
        top_indices = similarities.argsort(descending=True)[:top_k]
        
        # Return top_k results with their scores and document data
        results = []
        for idx in top_indices:
            results.append({
                'id': self.documents[idx]['id'],
                'content': self.documents[idx]['content'],
                'score': float(similarities[idx])
            })
        
        return results

    def get_cache_stats(self) -> Dict[str, any]:
        """
        Return statistics about the cache performance.

        Returns:
            Dictionary with cache hit/miss statistics
        """
        # Calculate total cache accesses
        total = self.cache_hits + self.cache_misses
        # Calculate hit rate percentage
        hit_rate_percent = (self.cache_hits / total * 100) if total > 0 else 0.0
        # Return a dictionary with hits, misses, total, and hit rate
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'total': total,
            'hit_rate_percent': hit_rate_percent
        }


# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        {"id": "doc1", "content": "How to reset your password in our application"},
        {"id": "doc2", "content": "Troubleshooting login issues and account access problems"},
        {"id": "doc3", "content": "Understanding your monthly billing statement"},
        {"id": "doc4", "content": "How to upgrade your subscription plan"},
        {"id": "doc5", "content": "Setting up two-factor authentication for security"},
    ]

    # Sample queries
    queries = [
        "I forgot my password",
        "Can't log into my account",
        "How do I understand my bill",
        "I want to upgrade my account",
        "password reset",
        "I forgot my password",  # Repeated query to test caching
    ]

    # Initialize and use the search engine
    search_engine = SemanticSearchEngine()

    # Add documents
    start_time = time.time()
    search_engine.add_documents(documents)
    print(f"Document processing time: {time.time() - start_time:.4f}s")

    # Search with each query
    for query in queries:
        start_time = time.time()
        results = search_engine.search(query)
        print(f"\nQuery: '{query}'")
        print(f"Search time: {time.time() - start_time:.4f}s")

        for result in results:
            print(
                f"  - {result['id']} (Score: {result['score']:.4f}): {result['content']}")

    # Print cache statistics
    print("\nCache statistics:")
    stats = search_engine.get_cache_stats()
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Total cache accesses: {stats['total']}")
    print(f"Hit rate: {stats['hit_rate_percent']:.2f}%")
