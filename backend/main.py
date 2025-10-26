from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import httpx
import json
from datetime import datetime, timedelta
import os

try:
    from advanced_rag import (
        SemanticChunker,
        HybridRetriever,
        SpacedRepetition,
        AdaptiveTestGenerator,
        PerformanceAnalyzer,
        WeaknessIdentifier,
        PromptOptimizer,
        ConceptExtractor,
        QuestionDifficultyEstimator
    )
    ADVANCED_FEATURES = True
except ImportError:
    print("⚠️ Advanced features not available. Using basic RAG only.")
    ADVANCED_FEATURES = False

app = FastAPI(title="StudyMate RAG API", version="2.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Initialize ChromaDB
print("🔧 Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Initialize embedding model
print(f"🔧 Loading embedding model: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)

# Create or get collection
collection = chroma_client.get_or_create_collection(
    name="study_notes",
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"}
)

# Initialize advanced RAG components
if ADVANCED_FEATURES:
    print("Initializing advanced RAG features...")
    semantic_chunker = SemanticChunker(model_name=EMBEDDING_MODEL_NAME)
    hybrid_retriever = HybridRetriever(collection, embedding_model)
    spaced_rep = SpacedRepetition()
    adaptive_tester = AdaptiveTestGenerator()
    performance_analyzer = PerformanceAnalyzer()
    weakness_identifier = WeaknessIdentifier()
    prompt_optimizer = PromptOptimizer()
    concept_extractor = ConceptExtractor()
    difficulty_estimator = QuestionDifficultyEstimator()
    print("Advanced features initialized!")

# Pydantic Models
class Note(BaseModel):
    title: str
    content: str
    subject: str
    review_days: int = 7
    use_semantic_chunking: bool = True

class NoteResponse(BaseModel):
    id: str
    title: str
    content: str
    subject: str
    date: str
    next_review: str
    key_concepts: Optional[List[dict]] = None
    chunks_created: int

class TestRequest(BaseModel):
    note_id: str
    num_questions: int = 5
    difficulty: str = "medium"
    use_adaptive: bool = False
    bloom_levels: Optional[List[str]] = None

class Answer(BaseModel):
    question_id: int
    answer: str

class TestSubmission(BaseModel):
    note_id: str
    answers: List[Answer]

class UserHistory(BaseModel):
    user_id: str
    test_history: List[dict]

# Helper Functions
async def call_ollama(prompt: str, system_prompt: str = "") -> str:
    """Call local Ollama instance"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "temperature": 0.7,
        "options": {
            "num_predict": 1024
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            return response.json()["response"]
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama request timed out. Try reducing num_questions.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

def chunk_text_basic(text: str, chunk_size: int = 500) -> List[str]:
    """Basic chunking by word count"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks if chunks else [text]

def retrieve_context(query: str, note_id: str, top_k: int = 5) -> str:
    """Retrieve context from ChromaDB"""
    try:
        if ADVANCED_FEATURES:
            # hybrid retrieval
            results = hybrid_retriever.retrieve(
                query=query,
                note_id=note_id,
                top_k=top_k,
                alpha=0.7
            )
            return "\n\n".join([r['document'] for r in results]) if results else ""
        else:
            # basic semantic search
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"note_id": note_id}
            )
            return "\n\n".join(results['documents'][0]) if results['documents'] else ""
    except Exception as e:
        print(f"Retrieval error: {str(e)}")
        return ""

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "StudyMate RAG API",
        "version": "2.0.0",
        "status": "running",
        "advanced_features": ADVANCED_FEATURES,
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "notes": "/api/notes",
            "test": "/api/generate-test",
            "evaluate": "/api/evaluate-test"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ollama": "checking...",
        "chromadb": "checking...",
        "advanced_features": ADVANCED_FEATURES
    }
    
    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            health_status["ollama"] = "connected"
            # Get available models
            models = response.json().get("models", [])
            health_status["ollama_models"] = [m["name"] for m in models]
    except:
        health_status["ollama"] = "disconnected"
        health_status["status"] = "degraded"
    
    # Check ChromaDB
    try:
        count = collection.count()
        health_status["chromadb"] = "connected"
        health_status["notes_count"] = count
    except Exception as e:
        health_status["chromadb"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"
    
    return health_status

@app.post("/api/notes", response_model=NoteResponse)
async def create_note(note: Note):
    """
    Create a new note with embeddings
    Supports both basic and semantic chunking
    """
    try:
        note_id = f"note_{int(datetime.now().timestamp() * 1000)}"
        
        # Extract key concepts if advanced features available
        key_concepts = None
        if ADVANCED_FEATURES:
            try:
                key_concepts = concept_extractor.extract_key_concepts(note.content, top_n=10)
            except Exception as e:
                print(f"Concept extraction failed: {str(e)}")
        
        # Choosing chunking method
        if ADVANCED_FEATURES and note.use_semantic_chunking:
            try:
                chunks = semantic_chunker.chunk_by_similarity(note.content, threshold=0.5)
            except Exception as e:
                print(f"⚠️ Semantic chunking failed, using basic: {str(e)}")
                chunks = chunk_text_basic(note.content)
        else:
            chunks = chunk_text_basic(note.content)
        
        # Store in ChromaDB
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                ids=[f"{note_id}_chunk_{i}"],
                metadatas=[{
                    "note_id": note_id,
                    "title": note.title,
                    "subject": note.subject,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }]
            )
        
        # Calculating next review date
        next_review = (datetime.now() + timedelta(days=note.review_days)).strftime("%Y-%m-%d")
        
        return NoteResponse(
            id=note_id,
            title=note.title,
            content=note.content,
            subject=note.subject,
            date=datetime.now().strftime("%Y-%m-%d"),
            next_review=next_review,
            key_concepts=key_concepts,
            chunks_created=len(chunks)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create note: {str(e)}")

@app.post("/api/generate-test")
async def generate_test(request: TestRequest):
    """
    Generate test questions using RAG + LLM
    Supports adaptive testing based on history
    """
    try:
        # Getting adaptive config if requested
        bloom_levels = request.bloom_levels or ['understand', 'apply', 'analyze']
        
        if ADVANCED_FEATURES and request.use_adaptive:
            # Would fetch real history from database
            student_history = []
            test_config = adaptive_tester.generate_adaptive_test(
                request.note_id,
                student_history
            )
            bloom_levels = test_config['bloom_levels']
        
        # Retrieve context
        context = retrieve_context(
            query=f"Generate {request.num_questions} questions about this topic",
            note_id=request.note_id,
            top_k=5
        )
        
        if not context:
            raise HTTPException(status_code=404, detail="Note not found or no content available")
        
        # Generate prompt
        if ADVANCED_FEATURES:
            prompt = prompt_optimizer.get_question_generation_prompt(
                context=context,
                num_questions=request.num_questions,
                difficulty=request.difficulty,
                bloom_levels=bloom_levels
            )
            system_prompt = "You are an expert educational content creator. Return ONLY valid JSON."
        else:
            system_prompt = "You are an expert educator. Return ONLY valid JSON with no extra text."
            prompt = f"""Based on this study material, generate {request.num_questions} test questions.

Study Material:
{context}

Generate a JSON array with this structure:
[
  {{
    "question": "Question text?",
    "type": "mcq",
    "options": ["A", "B", "C", "D"],
    "correct_answer": "A",
    "explanation": "Why this is correct"
  }},
  {{
    "question": "Short answer question?",
    "type": "short",
    "correct_answer": "Expected answer",
    "explanation": "Key points"
  }}
]

Include multiple choice and short answer questions."""
        
        # Call LLM
        response = await call_ollama(prompt, system_prompt)
        
        # Parse JSON
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join([l for l in lines if not l.strip().startswith("```")])
            if response.strip().startswith("json"):
                response = response.strip()[4:]
        
        questions = json.loads(response)
        
        return {
            "note_id": request.note_id,
            "questions": questions,
            "generated_at": datetime.now().isoformat(),
            "num_questions": len(questions),
            "bloom_levels": bloom_levels
        }
    
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse LLM response. Try again or reduce num_questions. Error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test generation failed: {str(e)}")

@app.post("/api/evaluate-test")
async def evaluate_test(submission: TestSubmission):
    """
    Evaluate student answers with detailed feedback
    Calculates spaced repetition schedule if available
    """
    try:
        context = retrieve_context(
            query="Evaluation context",
            note_id=submission.note_id,
            top_k=5
        )
        
        results = []
        total_score = 0
        
        for answer in submission.answers:
            # Generate evaluation prompt
            if ADVANCED_FEATURES:
                prompt = prompt_optimizer.get_answer_evaluation_prompt(
                    question=f"Question {answer.question_id}",
                    student_answer=answer.answer,
                    reference_answer="",
                    context=context
                )
                system_prompt = "You are an expert educator. Return ONLY valid JSON."
            else:
                system_prompt = "You are an expert educator evaluating answers. Return ONLY valid JSON."
                prompt = f"""Evaluate this answer based on the study material.

Study Material:
{context}

Student Answer: {answer.answer}

Return JSON:
{{
  "is_correct": true/false,
  "score": 0-100,
  "feedback": "Detailed constructive feedback",
  "key_points_covered": ["point1", "point2"],
  "improvement_suggestions": ["suggestion1"]
}}"""
            
            response = await call_ollama(prompt, system_prompt)
            
            try:
                response = response.strip()
                if response.startswith("```"):
                    lines = response.split("\n")
                    response = "\n".join([l for l in lines if not l.strip().startswith("```")])
                    if response.strip().startswith("json"):
                        response = response.strip()[4:]
                
                evaluation = json.loads(response)
                evaluation["question_id"] = answer.question_id
                results.append(evaluation)
                total_score += evaluation.get("score", 0)
            
            except json.JSONDecodeError:
                results.append({
                    "question_id": answer.question_id,
                    "is_correct": False,
                    "score": 50,
                    "feedback": "Could not evaluate answer. Please try again.",
                    "key_points_covered": [],
                    "improvement_suggestions": []
                })
                total_score += 50
        
        avg_score = total_score / len(submission.answers) if submission.answers else 0
        
        # Calculate spaced repetition if available
        spaced_rep_data = None
        if ADVANCED_FEATURES:
            quality = spaced_rep.quality_from_score(avg_score)
            next_interval, new_ef, new_reps = spaced_rep.calculate_next_interval(
                quality=quality,
                interval=7,
                easiness_factor=2.5,
                repetitions=0
            )
            spaced_rep_data = {
                "next_review_days": next_interval,
                "next_review_date": (datetime.now() + timedelta(days=next_interval)).strftime("%Y-%m-%d"),
                "quality_rating": quality,
                "easiness_factor": round(new_ef, 2),
                "repetitions": new_reps
            }
        
        return {
            "overall_score": round(avg_score, 2),
            "total_questions": len(submission.answers),
            "results": results,
            "evaluated_at": datetime.now().isoformat(),
            "spaced_repetition": spaced_rep_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/api/notes/{note_id}/context")
async def get_note_context(note_id: str, query: Optional[str] = None):
    """Retrieve note content or query-specific context"""
    try:
        if query:
            context = retrieve_context(query, note_id)
        else:
            results = collection.get(where={"note_id": note_id})
            context = "\n\n".join(results['documents']) if results['documents'] else ""
        
        if not context:
            raise HTTPException(status_code=404, detail="Note not found")
        
        return {
            "note_id": note_id,
            "context": context,
            "retrieved_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Advanced endpoints (only if features available)
if ADVANCED_FEATURES:
    
    @app.post("/api/analytics/performance")
    async def get_performance_analytics(history: UserHistory):
        """Generate comprehensive performance analytics"""
        try:
            analytics = performance_analyzer.generate_analytics(history.test_history)
            return {
                "user_id": history.user_id,
                "analytics": analytics,
                "generated_at": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/analytics/weaknesses")
    async def identify_weaknesses(history: UserHistory):
        """Identify weak topics and provide recommendations"""
        try:
            weak_topics = weakness_identifier.identify_weak_topics(
                history.test_history,
                threshold=0.7
            )
            return {
                "user_id": history.user_id,
                "weak_topics": weak_topics,
                "analyzed_at": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/notes/{note_id}/concepts")
    async def get_note_concepts(note_id: str):
        """Extract key concepts from a note"""
        try:
            results = collection.get(where={"note_id": note_id})
            if not results['documents']:
                raise HTTPException(status_code=404, detail="Note not found")
            
            content = "\n\n".join(results['documents'])
            key_concepts = concept_extractor.extract_key_concepts(content, top_n=15)
            concept_map = concept_extractor.generate_concept_map(key_concepts)
            
            return {
                "note_id": note_id,
                "key_concepts": key_concepts,
                "concept_map": concept_map
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Startup event
@app.on_event("startup")
async def startup_event():
    print("\n" + "="*60)
    print("StudyMate RAG API Starting...")
    print("="*60)
    print(f"ChromaDB Path: {CHROMA_DB_PATH}")
    print(f"Ollama Model: {OLLAMA_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"⚡ Advanced Features: {'Enabled' if ADVANCED_FEATURES else 'Disabled'}")
    print("="*60)
    print("Server ready! Visit http://localhost:8000/docs for API docs")
    print("="*60 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")