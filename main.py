import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union, Tuple, List, Optional
from io import BytesIO
import time
import json
import pickle
from pathlib import Path

# Vector DB imports (optional)
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Constants
DB_PATH = "db"
VECTOR_DB_PATH = "vector_db"
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
FEATURE_DIM = 2048  # ResNet50 feature dimension

# Global model cache
@st.cache_resource
def load_model() -> tf.keras.Model:
    """Load pre-trained ResNet50 model."""
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')

class VectorDB:
    """Vector database abstraction layer."""
    
    def __init__(self, db_type: str = "chroma", persist_dir: str = VECTOR_DB_PATH):
        self.db_type = db_type
        self.persist_dir = persist_dir
        self.collection = None
        self.index = None
        self.metadata = {}
        
        if db_type == "chroma" and CHROMA_AVAILABLE:
            self._init_chroma()
        elif db_type == "faiss" and FAISS_AVAILABLE:
            self._init_faiss()
        else:
            self._init_file_based()
    
    def _init_chroma(self):
        """Initialize ChromaDB."""
        client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = client.get_or_create_collection(
            name="image_features",
            metadata={"hnsw:space": "cosine"}
        )
    
    def _init_faiss(self):
        """Initialize FAISS index."""
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index_path = os.path.join(self.persist_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(FEATURE_DIM)  # Inner product for cosine similarity
    
    def _init_file_based(self):
        """Fallback to file-based storage."""
        os.makedirs(self.persist_dir, exist_ok=True)
        self.features_path = os.path.join(self.persist_dir, "features.npy")
        self.metadata_path = os.path.join(self.persist_dir, "metadata.json")
        
        if os.path.exists(self.features_path):
            self.features = np.load(self.features_path)
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.features = np.array([]).reshape(0, FEATURE_DIM)
            self.metadata = {"paths": [], "ids": []}
    
    def add_vectors(self, vectors: np.ndarray, paths: List[str], ids: Optional[List[str]] = None):
        """Add vectors to database."""
        if ids is None:
            ids = [f"img_{i}" for i in range(len(vectors))]
        
        if self.db_type == "chroma" and self.collection:
            # Convert to list format for ChromaDB
            embeddings = vectors.tolist()
            metadatas = [{"path": path} for path in paths]
            
            # Add with unique IDs to avoid duplicates
            try:
                self.collection.add(
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                st.error(f"ChromaDB add error: {e}")
                return False
        
        elif self.db_type == "faiss" and self.index:
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vectors)
            self.index.add(vectors)
            self.metadata["paths"].extend(paths)
            self.metadata["ids"].extend(ids)
            self._save_faiss()
        
        else:  # File-based
            if len(self.features) == 0:
                self.features = vectors
            else:
                self.features = np.vstack([self.features, vectors])
            self.metadata["paths"].extend(paths)
            self.metadata["ids"].extend(ids)
            self._save_file_based()
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors."""
        if self.db_type == "chroma" and self.collection:
            try:
                # Convert query vector to list format
                query_embedding = query_vector.tolist()
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, self.collection.count())
                )
                
                # ChromaDB returns distances (lower = more similar)
                # Convert to similarity scores (higher = more similar)
                if results["metadatas"] and results["metadatas"][0]:
                    similarities = []
                    for i, (meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
                        # Convert distance to similarity (1 - distance for cosine distance)
                        similarity = 1.0 - dist
                        similarities.append((meta["path"], similarity))
                    return similarities
                else:
                    return []
            except Exception as e:
                st.error(f"ChromaDB search error: {e}")
                return []
        
        elif self.db_type == "faiss" and self.index:
            if self.index.ntotal == 0:
                return []
            faiss.normalize_L2(query_vector.reshape(1, -1))
            distances, indices = self.index.search(query_vector.reshape(1, -1), top_k)
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata["paths"]):
                    results.append((self.metadata["paths"][idx], float(dist)))
            return results
        
        else:  # File-based
            if len(self.features) == 0:
                return []
            similarities = cosine_similarity([query_vector], self.features)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [(self.metadata["paths"][i], similarities[i]) 
                   for i in top_indices if similarities[i] > 0]
    
    def _save_faiss(self):
        """Save FAISS index and metadata."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def _save_file_based(self):
        """Save file-based data."""
        np.save(self.features_path, self.features)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def clear(self):
        """Clear all data."""
        if self.db_type == "chroma" and self.collection:
            try:
                # Delete the collection and recreate it
                client = chromadb.PersistentClient(path=self.persist_dir)
                try:
                    client.delete_collection("image_features")
                except:
                    pass  # Collection might not exist
                # Recreate collection
                self.collection = client.get_or_create_collection(
                    name="image_features",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                st.error(f"ChromaDB clear error: {e}")
        elif self.db_type == "faiss":
            self.index = faiss.IndexFlatIP(FEATURE_DIM)
            self.metadata = {"paths": [], "ids": []}
            self._save_faiss()
        else:
            self.features = np.array([]).reshape(0, FEATURE_DIM)
            self.metadata = {"paths": [], "ids": []}
            self._save_file_based()
    
    def get_count(self) -> int:
        """Get number of vectors in database."""
        if self.db_type == "chroma" and self.collection:
            return self.collection.count()
        elif self.db_type == "faiss" and self.index:
            return self.index.ntotal
        else:
            return len(self.metadata["paths"])

class ImageSearchEngine:
    def __init__(self, db_path: str, vector_db_type: str = "chroma"):
        self.db_path = db_path
        self.model = load_model()
        self.vector_db = VectorDB(vector_db_type)
        self.last_update = 0
        
    def extract_features(self, image_input: Union[str, BytesIO]) -> np.ndarray:
        """Extract features from image."""
        try:
            img = image.load_img(image_input, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = self.model.predict(img_array, verbose=0).flatten()
            return features
        except Exception as e:
            st.error(f"Feature extraction failed: {e}")
            return None

    def get_db_images(self) -> List[str]:
        """Get all image files from database."""
        if not os.path.exists(self.db_path):
            return []
        
        images = []
        for file in os.listdir(self.db_path):
            if file.lower().endswith(SUPPORTED_FORMATS):
                images.append(os.path.join(self.db_path, file))
        return sorted(images)
    
    def load_database(self, force_reload: bool = False) -> bool:
        """Load or reload database features into vector DB."""
        current_images = self.get_db_images()
        
        if not force_reload and self.vector_db.get_count() > 0:
            if len(current_images) == self.vector_db.get_count():
                return True
        
        if not current_images:
            st.warning("No images found in database!")
            return False
        
        with st.spinner(f"Loading {len(current_images)} images into vector DB..."):
            features = []
            valid_paths = []
            
            progress_bar = st.progress(0)
            for i, img_path in enumerate(current_images):
                feature = self.extract_features(img_path)
                if feature is not None:
                    features.append(feature)
                    valid_paths.append(img_path)
                progress_bar.progress((i + 1) / len(current_images))
            
            if features:
                # Clear existing data if force reload
                if force_reload:
                    self.vector_db.clear()
                
                # Add to vector database with unique IDs
                feature_matrix = np.vstack(features)
                unique_ids = [f"img_{hash(path)}_{i}" for i, path in enumerate(valid_paths)]
                self.vector_db.add_vectors(feature_matrix, valid_paths, unique_ids)
                self.last_update = time.time()
                progress_bar.empty()
                return True
            else:
                st.error("Failed to extract features from any images!")
                return False
    
    def find_similar(self, query_image: Union[str, BytesIO], 
                    threshold: float = 0.5, top_n: int = 5) -> List[Tuple[str, float]]:
        """Find similar images using vector database."""
        if self.vector_db.get_count() == 0:
            return []
        
        query_features = self.extract_features(query_image)
        if query_features is None:
            return []

        # Search vector database
        results = self.vector_db.search(query_features, top_n)
        
        # Filter by threshold
        filtered_results = [(path, score) for path, score in results if score > threshold]
        return filtered_results
    
    def add_image_to_db(self, image_file: BytesIO, filename: str) -> bool:
        """Add uploaded image to database."""
        try:
            file_path = os.path.join(self.db_path, filename)
            with open(file_path, "wb") as f:
                f.write(image_file.getbuffer())
            return True
        except Exception as e:
            st.error(f"Failed to save image: {e}")
            return False

def main():
    st.set_page_config(page_title="Visual Search Engine", layout="wide")
    st.title("🔍 Visual Image Search Engine")
    
    # Vector DB selection
    with st.sidebar:
        st.header("Vector Database")
        vector_db_options = ["file_based"]
        if CHROMA_AVAILABLE:
            vector_db_options.append("chroma")
        if FAISS_AVAILABLE:
            vector_db_options.append("faiss")
        
        vector_db_type = st.selectbox("Vector DB Type", vector_db_options, 
                                    help="Chroma: Best for production, FAISS: Fastest, File: Fallback")
    
    # Initialize search engine
    if "search_engine" not in st.session_state or st.session_state.get("vector_db_type") != vector_db_type:
        st.session_state.search_engine = ImageSearchEngine(DB_PATH, vector_db_type)
        st.session_state.vector_db_type = vector_db_type
    
    search_engine = st.session_state.search_engine
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        # Database management
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh DB", help="Reload database"):
                search_engine.load_database(force_reload=True)
                st.rerun()
        
        with col2:
            if st.button("📊 DB Stats"):
                images = search_engine.get_db_images()
                st.write(f"**Images in DB:** {len(images)}")
                st.write(f"**Vector DB:** {search_engine.vector_db.get_count()} vectors")
                st.write(f"**DB Type:** {vector_db_type}")
                
                # Debug info for ChromaDB
                if vector_db_type == "chroma" and search_engine.vector_db.collection:
                    try:
                        count = search_engine.vector_db.collection.count()
                        st.write(f"**ChromaDB Count:** {count}")
                        if count > 0:
                            # Get a sample to verify data
                            sample = search_engine.vector_db.collection.get(limit=1)
                            if sample and sample.get('metadatas'):
                                st.write(f"**Sample Path:** {sample['metadatas'][0]['path']}")
                    except Exception as e:
                        st.write(f"**ChromaDB Error:** {e}")
        
        # Search parameters
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.01)
        top_n = st.slider("Max Results", 1, 20, 5)
        
        # Image upload to database
        st.header("Add to Database")
        uploaded_db_image = st.file_uploader("Upload Image to DB", 
                                           type=['jpg', 'jpeg', 'png'], 
                                           key="db_upload")
        
        if uploaded_db_image and st.button("Add to DB"):
            if search_engine.add_image_to_db(uploaded_db_image, uploaded_db_image.name):
                st.success(f"Added {uploaded_db_image.name} to database!")
                search_engine.load_database(force_reload=True)
                st.rerun()
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Query Image")
        query_image = st.file_uploader("Choose query image...", 
                                      type=['jpg', 'jpeg', 'png'],
                                      key="query_upload")
        
        if query_image:
            img = Image.open(query_image)
            st.image(img, caption="Query Image", use_column_width=True)
            
            if st.button("🔍 Search", type="primary"):
                # Load database if needed
                if not search_engine.load_database():
                    st.error("Database loading failed!")
                return

                # Find similar images
                with st.spinner("Searching..."):
                    results = search_engine.find_similar(query_image, threshold, top_n)
                
                # Debug info
                if vector_db_type == "chroma":
                    st.write(f"**Debug - Raw results count:** {len(results)}")
                    if results:
                        st.write(f"**Debug - First result:** {results[0]}")
                
                if results:
                    st.success(f"Found {len(results)} similar images!")
                    
                    with col2:
                        st.header("Similar Images")
                        cols = st.columns(2)
                        
                        for i, (img_path, score) in enumerate(results):
                            with cols[i % 2]:
                                try:
                                    img = Image.open(img_path)
                                    st.image(img, caption=f"Score: {score:.3f}", use_column_width=True)
                                    st.write(f"**{os.path.basename(img_path)}**")
                                except Exception as e:
                                    st.error(f"Error loading {img_path}: {e}")
                else:
                    st.warning("No similar images found!")

    # Database status
    if search_engine.vector_db.get_count() > 0:
        st.info(f"✅ Vector DB loaded: {search_engine.vector_db.get_count()} images ready for search")

if __name__ == "__main__":
    main()