"""
🔥 REAL RAG SYSTEM - Complete Learning Implementation
=====================================================

This is a TRUE Retrieval-Augmented Generation system with:
1. Vector embeddings (semantic understanding)
2. Vector database (FAISS)
3. Document chunking
4. Semantic similarity search
5. Sourced answer generation

NOT just keyword matching - ACTUAL RAG!
"""

# ============= WHAT IS RAG? (EXPLAINED) =============

"""
TRADITIONAL LLM (Without RAG):
================================
User: "What is CO2 suppression?"
         ↓
    LLM only has training data
    (memorized from internet)
         ↓
    "CO2 is... uh... some gas... maybe used for fire?"
         ↓
    Problem: Hallucinations, outdated info, no sources

RAG SYSTEM (With Retrieval):
============================
User: "What is CO2 suppression?"
         ↓
    [STEP 1] RETRIEVE
    Search vector database for "CO2 suppression" docs
    ↓
    Return top 3 most SIMILAR documents
    (using cosine similarity of embeddings)
    ↓
    [STEP 2] AUGMENT
    Add these documents to the prompt:
    "Here are relevant documents:
     <Document 1: NFPA 13 CO2 Systems>
     <Document 2: Server Room Suppression>
     <Document 3: CO2 Safety Procedures>
     
     Based on these documents, answer: What is CO2 suppression?"
    ↓
    [STEP 3] GENERATE
    LLM reads the documents (not just memory)
    ↓
    "Based on NFPA 13: CO2 suppression is a clean agent system
     that displaces oxygen to extinguish fires. It uses 34% CO2
     concentration and is ideal for server rooms..."
    
    ✅ ADVANTAGES:
       - Factual (based on documents)
       - Sourced (can cite references)
       - Up-to-date (uses latest documents)
       - No hallucinations
"""

# ============= KEY CONCEPTS =============

"""
1. EMBEDDINGS (Converting text to vectors):
   
   Traditional NLP: "hello world" → [token1, token2]
   
   Embeddings: "hello world" → [0.23, -0.15, 0.89, ..., 0.42]
              (768 dimensions for good models)
   
   Semantic space: Similar texts have similar vectors
   - "CO2 suppression" similar to "Carbon dioxide fire system"
   - "Evacuation procedure" different from "Sprinkler activation"

2. VECTOR DATABASE (FAISS - Facebook AI Similarity Search):
   
   Traditional DB:
   SELECT * FROM docs WHERE keyword = "CO2"
   
   Vector DB:
   SEARCH vector_index FOR embedding_of_query
   RETURN top_k most_similar embeddings
   
   Speed: 100,000 documents in < 1ms

3. SIMILARITY METRICS:
   
   Cosine Similarity: Measures angle between vectors
   - Range: -1 to 1
   - 1 = identical
   - 0 = orthogonal (unrelated)
   - -1 = opposite
   
   Example:
   "CO2 suppression" vs "Carbon dioxide fire system": 0.92 (very similar)
   "CO2 suppression" vs "Lobby furniture": 0.15 (not similar)

4. CHUNKING (Breaking documents into pieces):
   
   Without chunking:
   Document: "NFPA 101 states... [10,000 words] ...end of document"
   Problem: Too long, loses focus
   
   With chunking (1000 tokens per chunk, 200 overlap):
   Chunk 1: "NFPA 101 states evacuation principles... [1000 tokens]"
   Chunk 2: "...principles apply to all buildings. Specifically... [1000 tokens]"
   Chunk 3: "...exit signs must be visible..." [1000 tokens]
   
   Benefit: Focused, retrievable pieces
"""

# ============= ARCHITECTURE =============

import json
from pathlib import Path
from typing import List, Dict, Tuple

# For REAL implementation, you need:
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("⚠️  Note: Run: pip install sentence-transformers faiss-cpu")
    print("   For actual RAG implementation with vector embeddings")


class RealRAGSystem:
    """
    TRUE RAG implementation with vector embeddings and semantic search
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG system
        
        model_name: Hugging Face model for embeddings
        - all-MiniLM-L6-v2: Fast, 384 dimensions (GOOD FOR LEARNING)
        - all-mpnet-base-v2: Better quality, 768 dimensions
        - all-roberta-large-v1: Best quality, 1024 dimensions
        
        Trade-off: Speed vs Quality
        """
        
        self.model_name = model_name
        self.embeddings_model = None
        self.vector_index = None  # FAISS index
        self.documents = []  # Store original documents
        self.document_embeddings = None  # numpy array of embeddings
        
        if DEPENDENCIES_AVAILABLE:
            print(f"📦 Loading embedding model: {model_name}")
            self.embeddings_model = SentenceTransformer(model_name)
            print(f"✅ Model loaded. Embedding dimension: 384")
            # Automatically load documents on init
            self.add_documents(create_fire_safety_documents())
        else:
            print("⚠️  Dependencies not available. Using mock implementation.")
    
    def chunk_document(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split long document into overlapping chunks
        
        Why: 
        - LLMs have token limits
        - Long documents lose focus
        - Overlapping helps continuity
        
        Example:
        Text: "NFPA 101 states... [1000 words]"
        Chunk 1: word 0-500
        Chunk 2: word 450-950  (overlap = 50 words)
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.split()) > 50:  # Only keep substantial chunks
                chunks.append(chunk)
        
        return chunks
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Add documents to RAG system
        
        Process:
        1. Chunk each document
        2. Generate embeddings for each chunk
        3. Build FAISS index
        4. Store original documents for retrieval
        
        Input format:
        [
            {"id": "doc1", "title": "...", "content": "..."},
            {"id": "doc2", "title": "...", "content": "..."}
        ]
        """
        
        print(f"📚 Processing {len(documents)} documents...")
        
        self.documents = []
        all_chunks = []
        
        # Step 1: Chunk documents
        for doc in documents:
            doc_id = doc.get("id", "unknown")
            title = doc.get("title", "")
            content = doc.get("content", "")
            
            chunks = self.chunk_document(content)
            print(f"   {doc_id}: {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                self.documents.append({
                    "doc_id": doc_id,
                    "title": title,
                    "chunk_id": i,
                    "content": chunk
                })
                all_chunks.append(chunk)
        
        print(f"✅ Total chunks: {len(all_chunks)}")
        
        # Step 2: Generate embeddings (if available)
        if self.embeddings_model:
            print("🧠 Generating embeddings...")
            embeddings = self.embeddings_model.encode(all_chunks, show_progress_bar=True)
            self.document_embeddings = embeddings
            
            # Step 3: Build FAISS index
            print("🔍 Building FAISS index...")
            dimension = embeddings.shape[1]
            self.vector_index = faiss.IndexFlatL2(dimension)
            self.vector_index.add(embeddings.astype(np.float32))
            print(f"✅ Index built with {self.vector_index.ntotal} vectors")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        SEMANTIC RETRIEVAL (THE KEY PART OF RAG!)
        
        Process:
        1. Embed the query (convert to same vector space as documents)
        2. Search FAISS index for most similar vectors
        3. Return top_k most similar documents
        
        Why semantic > keyword:
        - "CO2 suppression" matches "carbon dioxide fire system"
        - Keyword matching would miss this
        - Understands meaning, not just words
        """
        
        if not self.embeddings_model or not self.vector_index:
            print("⚠️  No embeddings available")
            return []
        
        # Step 1: Embed the query
        query_embedding = self.embeddings_model.encode([query])[0]
        
        # Step 2: Search FAISS
        distances, indices = self.vector_index.search(
            np.array([query_embedding]).astype(np.float32),
            top_k
        )
        
        # Step 3: Build results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                doc_info = self.documents[idx]
                doc_info["similarity_score"] = 1 / (1 + distance)  # Convert to similarity
                results.append(doc_info)
        
        return results
    
    def get_zone_procedure(self, zone_id: int) -> dict:
        """Get procedure for specific zone using semantic search"""
        zone_map = {1: "Lobby", 2: "Server Room", 3: "Warehouse"}
        zone_name = zone_map.get(zone_id, "Unknown")
        results = self.retrieve(f"{zone_name} fire response protocol evacuation", top_k=1)
        if results:
            return {
                "zone_id": zone_id,
                "source": results[0]["title"],
                "category": f"Zone: {zone_name}",
                "content": results[0]["content"]
            }
        return None
    
    def get_suppression_info(self, zone_id: int) -> dict:
        """Get suppression system info for zone using semantic search"""
        zone_suppressions = {1: "Sprinkler", 2: "CO2", 3: "Foam"}
        suppression_type = zone_suppressions.get(zone_id, "Unknown")
        results = self.retrieve(f"{suppression_type} suppression system response time", top_k=1)
        if results:
            return {
                "zone_id": zone_id,
                "suppression_type": suppression_type,
                "info": results[0]
            }
        return None
    
    def augment_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """
        AUGMENTATION (Adding context to prompt)
        
        This is what makes RAG work:
        Instead of:
            "What is CO2 suppression?"
        
        We ask LLM:
            "Here are some documents:
             <doc1>: CO2 suppression is...
             <doc2>: Server room uses...
             
             Based on these documents, answer: What is CO2 suppression?"
        """
        
        prompt = f"""Based on the following documents, answer the question:

DOCUMENTS:
"""
        for i, doc in enumerate(context_docs, 1):
            prompt += f"\n[Document {i}] {doc['title']} (Relevance: {doc.get('similarity_score', 0):.2f})\n"
            prompt += f"{doc['content'][:500]}...\n"
        
        prompt += f"\nQUESTION: {query}\n\nANSWER:"
        
        return prompt
    
    def generate_answer(self, query: str, llm_function=None) -> Dict:
        """
        FULL RAG PIPELINE
        
        1. Retrieve relevant documents (semantic search)
        2. Augment prompt with documents
        3. Generate answer using LLM
        """
        
        # Step 1: Retrieve
        print(f"\n🔍 Retrieving documents for: '{query}'")
        relevant_docs = self.retrieve(query, top_k=3)
        
        if not relevant_docs:
            return {"answer": "No relevant documents found", "sources": []}
        
        # Show what was retrieved
        for doc in relevant_docs:
            print(f"   ✅ {doc['title']} (score: {doc.get('similarity_score', 0):.3f})")
        
        # Step 2: Augment
        augmented_prompt = self.augment_prompt(query, relevant_docs)
        
        # Step 3: Generate (if LLM function provided)
        if llm_function:
            answer = llm_function(augmented_prompt)
        else:
            answer = f"[LLM would answer based on: {relevant_docs[0]['title']}]"
        
        return {
            "query": query,
            "answer": answer,
            "sources": [{"doc": d["title"], "chunk": d["chunk_id"]} for d in relevant_docs]
        }


# ============= EXAMPLE: FIRE SAFETY RAG =============

def create_fire_safety_documents() -> List[Dict[str, str]]:
    """
    Create proper documents for RAG
    (In real scenario, you'd download PDFs and extract text)
    """
    
    return [
        {
            "id": "nfpa_101",
            "title": "NFPA 101 Life Safety Code - Evacuation",
            "content": """
NFPA 101 Life Safety Code establishes fundamental requirements for the 
protection of life from fire and building hazards. 

EVACUATION REQUIREMENTS:
1. All occupants must be able to evacuate in orderly manner
2. Exit routes must be clearly marked and unobstructed
3. Exit signs must be illuminated and visible from any direction
4. Maximum travel distance to exit: 250 feet (standard), 200 feet (high hazard)

EVACUATION TIME TARGETS:
- Small buildings (< 5,000 sqft): 5-10 minutes
- Medium buildings (5,000-20,000 sqft): 10-15 minutes  
- Large buildings (> 20,000 sqft): 15-20 minutes
- Server/critical rooms: 2-3 minutes

EXIT ROUTE STANDARDS:
- Minimum width: 44 inches (single file line)
- Minimum ceiling height: 7 feet 6 inches
- No obstructions allowed
- Adequate lighting required (minimum 1 foot-candle)
- Emergency lighting required in darkness
            """
        },
        {
            "id": "nfpa_13",
            "title": "NFPA 13 Sprinkler Systems - Types and Response",
            "content": """
NFPA 13 Standard for Installation of Sprinkler Systems

SPRINKLER SYSTEM TYPES:

1. WET PIPE SPRINKLER SYSTEMS:
   Water is constantly present in pipes
   Fusible link melts → water sprays immediately
   Response time: < 60 seconds
   Coverage: Typical office, retail, warehouse areas
   Best for: Lobby (Warm climate areas)
   
2. DRY PIPE SPRINKLER SYSTEMS:
   Pressurized air holds back water
   Heat triggers air release → water follows
   Response time: 30-60 seconds
   Best for: Freezing environments, unheated areas
   
3. CO2 (CARBON DIOXIDE) SUPPRESSION:
   Displaces oxygen, extinguishes fire by suffocation
   Response time: 10-15 seconds (FASTEST)
   Concentration: 34% CO2 for Class B/C fires
   Safety: Requires automatic evacuation alert
   Advantages: Non-corrosive, no residue, safe for electronics
   Best for: Server rooms, data centers (CRITICAL INFRASTRUCTURE)
   
4. FOAM SUPPRESSION SYSTEMS:
   Foam blanket suppresses flammable liquid fires
   Activation: Manual or automatic with liquid detection
   Response time: 30-90 seconds
   Best for: Warehouse (flammable storage areas)

RESPONSE TIME COMPARISON:
- CO2: 10-15 seconds (FASTEST, safest for equipment)
- Wet Sprinkler: < 60 seconds
- Dry Sprinkler: 30-60 seconds
- Foam: 30-90 seconds
            """
        },
        {
            "id": "server_room_proc",
            "title": "Server Room Fire Response - Critical Infrastructure",
            "content": """
SERVER ROOM FIRE RESPONSE PROTOCOL

Zone: Server Room (Critical Infrastructure)
Occupancy: 2-5 people maximum
Area: 200 sq meters
Primary Hazard: Electrical fire, CO2 suppression danger
Suppression: CO2 System (CLEAN AGENT - NO WATER DAMAGE)
Evacuation Time Target: 2 minutes (CRITICAL)

IMMEDIATE RESPONSE (0-30 seconds):
1. EVACUATE IMMEDIATELY (do not suppress manually)
2. Activate nearest fire alarm
3. Announce: "EVACUATE SERVER ROOM - FIRE ALARM"
4. Close fireproof doors behind you (contain fire)
5. Do NOT attempt to shut down equipment

CO2 SUPPRESSION ACTIVATION (30-60 seconds):
- System floods room with CO2
- 34% CO2 concentration displaces oxygen
- Fire is extinguished by oxygen depletion
- Area becomes HAZARDOUS TO HUMANS
- Sound strobe alarm to warn occupants

OCCUPANT EVACUATION (0-2 minutes):
- All personnel must evacuate immediately
- CO2 is hazardous to human life
- Use nearest safe exit
- Do NOT re-enter for any reason
- Report to assembly point

CRITICAL CONSIDERATIONS:
- Server room should be MANNED only during business hours
- CO2 system is AUTOMATIC (no manual intervention)
- Room should be isolated from general HVAC
- Emergency doors must allow rapid exit
- Fire detection should be dual-sensor (smoke + heat)

POST-INCIDENT:
- Minimum 30 minutes before re-entry
- Allow full ventilation
- Document equipment damage
- Full inspection before restart
            """
        }
    ]


# ============= DEMO =============

def demo_real_rag():
    """
    Demonstrate REAL RAG system
    """
    
    print("\n" + "=" * 80)
    print("🔥 REAL RAG SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    if not DEPENDENCIES_AVAILABLE:
        print("\n⚠️  To run actual RAG, install dependencies:")
        print("   pip install sentence-transformers faiss-cpu")
        print("\n📚 But here's how it WOULD work:")
    
    # Create RAG system
    print("\n1️⃣  INITIALIZING RAG SYSTEM")
    print("-" * 80)
    rag = RealRAGSystem()
    
    # Add documents
    print("\n2️⃣  ADDING DOCUMENTS")
    print("-" * 80)
    documents = create_fire_safety_documents()
    rag.add_documents(documents)
    
    # Test queries
    print("\n3️⃣  SEMANTIC RETRIEVAL TESTS")
    print("-" * 80)
    
    queries = [
        "What suppression system is best for server rooms?",
        "How long does evacuation take?",
        "Tell me about CO2 suppression",
        "What's the response time for different sprinkler types?"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        if DEPENDENCIES_AVAILABLE:
            results = rag.retrieve(query, top_k=2)
            for i, doc in enumerate(results, 1):
                print(f"   [{i}] {doc['title']}")
                print(f"       Similarity: {doc.get('similarity_score', 0):.3f}")
                print(f"       Preview: {doc['content'][:100]}...")
        else:
            print("   [Would retrieve and rank documents by semantic similarity]")
    
    # Explain the difference
    print("\n4️⃣  WHY REAL RAG > KEYWORD MATCHING")
    print("-" * 80)
    print("""
    Query: "What's the fastest fire suppression system?"
    
    KEYWORD MATCHING (❌ WRONG):
    - Look for documents containing "fastest"
    - Miss documents about CO2 (which IS fastest but don't say "fastest")
    - Return: Nothing or irrelevant results
    
    REAL RAG (✅ CORRECT):
    - Embed query: "fastest fire suppression" → vector [0.23, -0.15, ...]
    - Find most similar document vectors
    - "CO2 response time 10-15 seconds" very similar to query
    - "Wet sprinkler < 60 seconds" also similar but less
    - Rank by semantic similarity
    - Return: CO2 first (correct answer!)
    """)


# ============= LEARNING CONCEPTS =============

learning_material = """
🎓 KEY LEARNING POINTS
=======================

1. EMBEDDINGS:
   - Text → Numbers (vectors)
   - Similar meaning → Similar vectors
   - Allows mathematical comparison

2. VECTOR DATABASES:
   - Fast nearest-neighbor search
   - FAISS: ~1ms for 100k documents
   - Essential for RAG at scale

3. CHUNKING:
   - Break documents into pieces
   - Overlapping chunks maintain context
   - Improves retrieval precision

4. SEMANTIC SEARCH:
   - Understand meaning, not just keywords
   - Cosine similarity for comparison
   - More robust than exact matching

5. AUGMENTED PROMPTING:
   - Add context to LLM prompt
   - LLM reads actual documents
   - Reduces hallucinations

6. WHY RAG MATTERS:
   - Base LLMs have outdated knowledge
   - RAG keeps information current
   - Sourced answers (auditability)
   - Works with proprietary data

7. AGENTIC RAG (What you chose!):
   - Agent DECIDES what to retrieve
   - Can call multiple tools
   - Combines retrieval + reasoning
   - State-of-the-art approach
"""

if __name__ == "__main__":
    demo_real_rag()
    
    print("\n" + "=" * 80)
    print(learning_material)
    print("=" * 80)
    
    print("\n📝 TO RUN REAL RAG:")
    print("   1. pip install sentence-transformers faiss-cpu")
    print("   2. python real_rag_system.py")
    print("\n🎯 NEXT: Integrate with Agentic system")
