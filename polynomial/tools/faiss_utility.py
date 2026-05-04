import os
import json
import time
import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class AnswerComparator:
    def __init__(self, output_dir:str, model_name: str ="all-MiniLM-L6-v2", cache_dir: str | None = None, reuse_existing: bool = False):
        """
        Initialize the answer comparator with output dir and a model.
        
        Args:
            output_dir: Directory to save FAISS index and metadata; if reuse_existing=True this is used directly
            model_name: Sentence transformer model to use for embeddings; can be a HF repo id or local path
            cache_dir: Optional cache directory for HF downloads to avoid unwritable defaults
            reuse_existing: If True, use output_dir as-is (no timestamp) so runs can share an index
        """
        if reuse_existing:
            self.output_dir = output_dir
        else:
            timestamp = str(int(time.time()))
            self.output_dir = os.path.join(output_dir, f"faiss_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.embeddings_file = os.path.join(self.output_dir, "embeddings.npy")
        self.embeddings = []
        self.metadata = []


        # Initialize embedding model
        try:
            # Use provided model name so callers can point to a local cache if needed
            self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        except OSError as exc:
            raise EnvironmentError(
                f"Could not load the sentence-transformer model '{model_name}'. "
                "If you are offline, download the model locally and pass its path via `model_name`. "
                "If the default HF cache is not writable, provide `cache_dir` pointing to a writable path."
            ) from exc
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        #    Load existing data if available
        self.load_existing_data()

    def load_existing_data(self):
        """Load existing embeddings and metadata from disk if they exist."""
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'r') as f:
                    data = json.load(f)
                    self.embeddings = data.get('embeddings', [])
                    self.metadata = data.get('metadata', [])
            except json.JSONDecodeError:
                backup_path = f"{self.embeddings_file}.corrupt.{int(time.time())}"
                try:
                    os.rename(self.embeddings_file, backup_path)
                    print(
                        f"Warning: FAISS embeddings file was corrupt; moved to {backup_path} and starting with an empty index."
                    )
                except OSError:
                    print(
                        "Warning: FAISS embeddings file was corrupt; could not move it. Starting with an empty index."
                    )
                self.embeddings = []
                self.metadata = []
                return

            # Convert embeddings back to numpy array
            if self.embeddings:
                self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild the FAISS index from the in-memory embeddings list."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        if not self.embeddings:
            return
        embeddings_np = np.array(self.embeddings).astype('float32')
        self.index.add(embeddings_np)
    
    def _save_data(self):
        """Save current embeddings and metadata to disk."""
        os.makedirs(self.output_dir, exist_ok=True)
        tmp_path = f"{self.embeddings_file}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump({
                'embeddings': self.embeddings,
                'metadata': self.metadata
            }, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.embeddings_file)

    def add_answer(self, answer: str, agent_name: str, round_num: int, run_id: str):
        """
        Add an answer to the index
        Args:
            answer: The answer text to embed and store
            agent_name: Name of the agent who provided the answer
            round_num: Round number of the answer
            run_id: ID of the run
        """
        embedding = self.model.encode(answer, convert_to_numpy=True)
        # Add to FAISS index
        self.index.add(np.array([embedding], dtype='float32'))

        # Store metadata
        self.embeddings.append(embedding.tolist())
        self.metadata.append({
            'answer': answer,
            'agent_name': agent_name,
            'round_num': round_num,
            'run_id': run_id
        })
        # Save to disk
        self._save_data()
    
    def remove_run(self, run_id: str) -> int:
        """
        Remove all embeddings/metadata associated with a given run_id.

        Returns:
            The number of entries removed.
        """
        kept_embeddings = []
        kept_metadata = []
        removed = 0
        for embedding, meta in zip(self.embeddings, self.metadata):
            if meta.get("run_id") == run_id:
                removed += 1
                continue
            kept_embeddings.append(embedding)
            kept_metadata.append(meta)

        if removed:
            self.embeddings = kept_embeddings
            self.metadata = kept_metadata
            self._save_data()
            self._rebuild_index()
        return removed


    def find_similar_answers(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar answers to a query using FAISS search.
        
        Args:
            query: The query text to search for
            top_k: Number of similar answers to return
            
        Returns:
            List of dictionaries containing similar answers and their metadata
        """
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32')
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)

        # Return results with metadata
        results = []
        for idx , score in zip(indices[0], distances[0]):
            if idx >= len(self.metadata):
                continue
            result = self.metadata[idx].copy() # Copy dict to avoid modifying original
            result['similarity_score'] = float(score) # Overwrite similarity score with FAISS distance
            results.append(result) 
        return results
                
    def compare_agent_answers(
        self,
        agent_name: str,
        round_num: int,
        k: int = 5,
        run_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Compare an agent's answers across different runs.

        Args:
            agent_name: Name of the agent.
            round_num: Round number to compare.
            k: Number of similar answers to return.
            run_id: If provided, use this run's answer as the query and exclude it from results.

        Returns:
            List of similar answers with metadata.
        """
        query = None
        if run_id:
            for meta in reversed(self.metadata):
                if (
                    meta.get("agent_name") == agent_name
                    and meta.get("round_num") == round_num
                    and meta.get("run_id") == run_id
                ):
                    query = meta.get("answer")
                    break
        if query is None:
            for meta in reversed(self.metadata):
                if meta.get("agent_name") == agent_name and meta.get("round_num") == round_num:
                    query = meta.get("answer")
                    break
        if query is None:
            for meta in reversed(self.metadata):
                if meta.get("agent_name") == agent_name:
                    query = meta.get("answer")
                    break

        if query is None:
            return []

        candidates = self.find_similar_answers(query, k + 1 if run_id else k)
        if run_id:
            candidates = [c for c in candidates if c.get("run_id") != run_id]
        return candidates[:k]
