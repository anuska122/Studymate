import os
from datetime import datetime, timedelta
from typing import List, Optional
import json
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

app = FastAPI(title="StudyMate RAG API with Ollama", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initializing ChromaDB

print("Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)
collection = chroma_client.get_or_create_collection(
    name="study_notes",
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"}
)
print("ChromaDB initialized successfully")

# Ollama Integration

def check_ollama_status():
    """Check if Ollama is running"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def call_ollama(prompt: str, max_tokens: int = 400) -> str:
    """Call Ollama API"""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Error: Ollama returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure it's running (ollama serve)"
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"

print(f"\nChecking Ollama connection at {OLLAMA_BASE_URL}...")
if check_ollama_status():
    print(f"✓ Ollama is running")
    print(f"✓ Using model: {OLLAMA_MODEL}")
else:
    print(f"⚠ WARNING: Ollama is not running!")
    print(f"Start it with: ollama serve")
    print(f"Then pull model: ollama pull {OLLAMA_MODEL}")

# Pydantic models

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

class Answer(BaseModel):
    question_id: int
    answer: str

class TestSubmission(BaseModel):
    note_id: str
    answers: List[Answer]

# Helper functions
def chunk_text_basic(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks if chunks else [text]

def retrieve_context(query: str, note_id: str, top_k: int = 5) -> str:
    """Retrieve relevant context from ChromaDB"""
    try:
        results = collection.query(
            query_texts=[query], 
            n_results=top_k, 
            where={"note_id": note_id}
        )
        if results['documents'] and results['documents'][0]:
            return "\n\n".join(results['documents'][0])
        return ""
    except Exception as e:
        print(f"Retrieval error: {e}")
        return ""

def extract_concepts(text: str, top_n: int = 10) -> List[dict]:
    """Extract key concepts using Ollama"""
    try:
        prompt = f"""Extract the {top_n} most important concepts from this text. 
For each concept, provide the concept name and a brief explanation.

Text:
{text[:1500]}

Format as JSON array: [{{"concept": "Name", "explanation": "Brief explanation"}}]"""
        
        response = call_ollama(prompt, max_tokens=300)
        
        # Try to parse JSON
        try:
            import re
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                concepts_raw = json.loads(json_match.group())
                concepts = []
                for i, c in enumerate(concepts_raw[:top_n]):
                    concepts.append({
                        'concept': c.get('concept', f'Concept {i+1}'),
                        'frequency': 1,
                        'example_context': c.get('explanation', '')[:100]
                    })
                return concepts
        except:
            pass
        
        # Fallback
        import re
        from collections import Counter
        concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        concept_counts = Counter(concepts)
        
        result = []
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        for concept, count in concept_counts.most_common(top_n):
            example = next((s for s in sentences if concept in s), "")
            result.append({
                'concept': concept,
                'frequency': count,
                'example_context': example[:100]
            })
        return result
    except Exception as e:
        print(f"Concept extraction error: {e}")
        return []

def generate_questions_with_ollama(context: str, num_questions: int = 5, difficulty: str = "medium") -> List[dict]:
    """Generate questions using Ollama"""
    try:
        prompt = f"""You are an expert educator. Create {num_questions} {difficulty} level test questions based on this study material.

Study Material:
{context[:2000]}

Generate exactly {num_questions} clear questions that test understanding.

Format as JSON array: [{{"question": "Your question?", "type": "text"}}]"""
        
        response = call_ollama(prompt, max_tokens=500)
        
        # Parse JSON
        try:
            import re
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                questions = json.loads(json_match.group())
                questions = questions[:num_questions]
                for idx, q in enumerate(questions):
                    q['question_id'] = idx
                    if 'type' not in q:
                        q['type'] = 'text'
                return questions
        except Exception as e:
            print(f"JSON parsing error: {e}")
        
        # Fallback
        sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 20]
        questions = []
        templates = [
            "What is {topic}?",
            "Explain {topic}.",
            "Describe {topic} in detail.",
            "How does {topic} work?",
            "What are the key aspects of {topic}?"
        ]
        
        for i, sentence in enumerate(sentences[:num_questions]):
            words = sentence.split()[:6]
            topic = ' '.join(words) if words else "this concept"
            questions.append({
                "question_id": i,
                "question": templates[i % len(templates)].format(topic=topic),
                "type": "text"
            })
        
        return questions[:num_questions]
    except Exception as e:
        print(f"Question generation error: {e}")
        return []

def evaluate_answer_with_ollama(question: str, student_answer: str, context: str) -> dict:
    """Evaluate student answer using Ollama"""
    try:
        prompt = f"""You are an expert educator. Evaluate this student's answer.

Question: {question}

Student's Answer:
{student_answer}

Reference Material:
{context[:1000]}

Provide: score (0-100), is_correct (true/false), and brief feedback.

Format as JSON: {{"score": 85, "is_correct": true, "feedback": "Good answer!"}}"""
        
        response = call_ollama(prompt, max_tokens=200)
        
        # Parse JSON
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "score": int(result.get("score", 50)),
                    "is_correct": result.get("is_correct", False),
                    "feedback": result.get("feedback", "Answer evaluated.")
                }
        except:
            pass
        
        # Fallback
        if not student_answer or len(student_answer.strip()) < 10:
            return {
                "score": 20,
                "is_correct": False,
                "feedback": "Answer is too short. Please provide more detail."
            }
        
        context_words = set(context.lower().split())
        answer_words = set(student_answer.lower().split())
        overlap = len(context_words & answer_words)
        score = min(100, int((overlap / max(len(context_words) * 0.3, 1)) * 100))
        
        return {
            "score": score,
            "is_correct": score >= 60,
            "feedback": "Good effort! " if score >= 60 else "Try to include more details."
        }
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {
            "score": 50,
            "is_correct": False,
            "feedback": "Unable to evaluate properly."
        }

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ollama_status = "connected" if check_ollama_status() else "disconnected"
    return {
        "status": "healthy",
        "ollama": ollama_status,
        "database": "connected",
        "model": OLLAMA_MODEL,
        "embedding_model": EMBEDDING_MODEL_NAME
    }

@app.post("/api/notes", response_model=NoteResponse)
async def create_note(note: Note):
    """Create a new note"""
    try:
        chunks = chunk_text_basic(note.content)
        existing_ids = collection.get()['ids']
        note_id = str(len(existing_ids) + 1)
        date_now = datetime.utcnow()
        next_review = date_now + timedelta(days=note.review_days)
        
        for idx, chunk in enumerate(chunks):
            collection.add(
                ids=[f"{note_id}_{idx}"],
                metadatas=[{
                    "note_id": note_id,
                    "chunk_idx": idx,
                    "title": note.title,
                    "subject": note.subject
                }],
                documents=[chunk]
            )
        
        print(f"Created note {note_id} with {len(chunks)} chunks")
        
        return NoteResponse(
            id=note_id,
            title=note.title,
            content=note.content,
            subject=note.subject,
            date=date_now.isoformat(),
            next_review=next_review.isoformat(),
            chunks_created=len(chunks)
        )
    except Exception as e:
        print(f"Error creating note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/notes/{note_id}/concepts")
async def get_note_concepts(note_id: str):
    """Extract key concepts"""
    try:
        results = collection.get(where={"note_id": note_id})
        
        if not results['documents']:
            raise HTTPException(status_code=404, detail="Note not found")
        
        full_text = ' '.join(results['documents'])
        key_concepts = extract_concepts(full_text)
        
        return {"key_concepts": key_concepts}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error extracting concepts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-test")
async def generate_test(req: TestRequest):
    """Generate test questions"""
    try:
        if not check_ollama_status():
            raise HTTPException(status_code=503, detail="Ollama is not running. Start it with: ollama serve")
        
        context = retrieve_context("test questions", req.note_id, top_k=10)
        
        if not context:
            raise HTTPException(status_code=404, detail="Note not found")
        
        questions = generate_questions_with_ollama(context, req.num_questions, req.difficulty)
        
        if not questions:
            raise HTTPException(status_code=500, detail="Failed to generate questions")
        
        return {
            "note_id": req.note_id,
            "questions": questions
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate-test")
async def evaluate_test(submission: TestSubmission):
    """Evaluate test answers"""
    try:
        context = retrieve_context("evaluation", submission.note_id, top_k=10)
        results = []
        
        for ans in submission.answers:
            eval_result = evaluate_answer_with_ollama(
                f"Question {ans.question_id}",
                ans.answer,
                context
            )
            eval_result['question_id'] = ans.question_id
            results.append(eval_result)
        
        overall_score = sum(r["score"] for r in results) / len(results) if results else 0
        
        if overall_score >= 80:
            days = 14
        elif overall_score >= 60:
            days = 7
        else:
            days = 3
        
        next_review = (datetime.utcnow() + timedelta(days=days)).isoformat()
        
        return {
            "overall_score": overall_score,
            "results": results,
            "spaced_repetition": {
                "next_review_date": next_review,
                "interval_days": days
            }
        }
    except Exception as e:
        print(f"Error evaluating test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "StudyMate RAG API with Ollama",
        "version": "2.0.0",
        "status": "running",
        "ollama_model": OLLAMA_MODEL
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("StudyMate Backend with Ollama")
    print("="*60)
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Embedding: {EMBEDDING_MODEL_NAME}")
    print(f"Database: {CHROMA_DB_PATH}")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")