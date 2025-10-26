# Advanced RAG techniques for StudyMate

from typing import List, Dict, Tuple
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# SEMANTIC CHUNKER

class SemanticChunker:
    """Split text into semantically meaningful chunks"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def chunk_by_similarity(self, text: str, threshold: float = 0.5, 
                           min_chunk_size: int = 100) -> List[str]:
        sentences = self._split_into_sentences(text)
        if not sentences or len(sentences) < 2:
            return [text]
        
        try:
            embeddings = self.model.encode(sentences)
            chunks = []
            current_chunk = [sentences[0]]
            
            for i in range(1, len(sentences)):
                similarity = cosine_similarity(
                    embeddings[i-1].reshape(1, -1),
                    embeddings[i].reshape(1, -1)
                )[0][0]
                
                if similarity < threshold or len(' '.join(current_chunk)) > 1000:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentences[i]]
                else:
                    current_chunk.append(sentences[i])
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return chunks if chunks else [text]
        except Exception as e:
            print(f"Semantic chunking error: {e}, falling back to text")
            return [text]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

# QUERY EXPANDER

class QueryExpander:
    """Expand user queries to improve retrieval"""
    
    def expand_with_synonyms(self, query: str) -> List[str]:
        synonyms = {
            'explain': ['describe', 'clarify', 'define'],
            'what is': ['define', 'explain', 'describe'],
            'how does': ['explain', 'describe the process'],
            'difference': ['compare', 'distinction', 'contrast']
        }
        
        queries = [query]
        for key, values in synonyms.items():
            if key in query.lower():
                for synonym in values:
                    queries.append(query.lower().replace(key, synonym))
        
        return queries
    
    def expand_with_context(self, query: str, note_subject: str) -> str:
        return f"{query} in the context of {note_subject}"

# ===================== HYBRID RETRIEVER =====================

class HybridRetriever:
    """Combine keyword-based and semantic search"""
    
    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model
    
    def retrieve(self, query: str, note_id: str, top_k: int = 5,
                 alpha: float = 0.5) -> List[Dict]:
        try:
            semantic_results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k * 2, 10),
                where={"note_id": note_id}
            )
            
            keyword_results = self._keyword_search(query, note_id, min(top_k * 2, 10))
            
            combined = self._fuse_results(semantic_results, keyword_results, alpha)
            
            return combined[:top_k]
        except Exception as e:
            print(f"Hybrid retrieval error: {e}")
            return []
    
    def _keyword_search(self, query: str, note_id: str, k: int) -> List[Dict]:
        try:
            results = self.collection.get(where={"note_id": note_id})
            
            if not results['documents']:
                return []
            
            query_terms = set(query.lower().split())
            scored_docs = []
            
            for i, doc in enumerate(results['documents']):
                doc_terms = set(doc.lower().split())
                score = len(query_terms & doc_terms) / max(len(query_terms), 1)
                scored_docs.append({
                    'document': doc,
                    'score': score,
                    'metadata': results['metadatas'][i] if results['metadatas'] else {}
                })
            
            return sorted(scored_docs, key=lambda x: x['score'], reverse=True)[:k]
        except Exception as e:
            print(f"Keyword search error: {e}")
            return []
    
    def _fuse_results(self, semantic: Dict, keyword: List[Dict], alpha: float) -> List[Dict]:
        scores = {}
        k = 60
        
        if semantic.get('documents') and semantic['documents']:
            for rank, doc in enumerate(semantic['documents'][0]):
                scores[doc] = scores.get(doc, 0) + alpha / (k + rank + 1)
        
        for rank, item in enumerate(keyword):
            doc = item['document']
            scores[doc] = scores.get(doc, 0) + (1 - alpha) / (k + rank + 1)
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{'document': doc, 'score': score} for doc, score in ranked]

# ===================== SPACED REPETITION =====================

class SpacedRepetition:
    """Anki-style SM-2 spaced repetition"""
    
    @staticmethod
    def calculate_next_interval(quality: int, interval: int, 
                               easiness_factor: float, repetitions: int) -> Tuple[int, float, int]:
        if quality >= 3:
            if repetitions == 0:
                interval = 1
            elif repetitions == 1:
                interval = 6
            else:
                interval = round(interval * easiness_factor)
            repetitions += 1
        else:
            repetitions = 0
            interval = 1
        
        easiness_factor = max(
            1.3,
            easiness_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        )
        
        return interval, easiness_factor, repetitions
    
    @staticmethod
    def quality_from_score(score: float) -> int:
        if score >= 90:
            return 5
        elif score >= 80:
            return 4
        elif score >= 60:
            return 3
        elif score >= 40:
            return 2
        elif score >= 20:
            return 1
        else:
            return 0

# ===================== QUESTION DIFFICULTY =====================

class QuestionDifficultyEstimator:
    """Estimate and calibrate question difficulty"""
    
    def estimate_difficulty(self, question: str, answer: str) -> str:
        factors = {
            'length': len(answer.split()) > 50,
            'requires_synthesis': any(word in question.lower() for word in ['compare', 'contrast', 'analyze', 'evaluate']),
            'factual': any(word in question.lower() for word in ['what is', 'define', 'list']),
            'multiple_concepts': len(re.findall(r'\band\b', question.lower())) > 1
        }
        
        difficulty_score = sum([
            factors['length'] * 2,
            factors['requires_synthesis'] * 3,
            -factors['factual'] * 2,
            factors['multiple_concepts'] * 2
        ])
        
        if difficulty_score >= 4:
            return 'hard'
        elif difficulty_score >= 2:
            return 'medium'
        else:
            return 'easy'
    
    def balance_question_set(self, questions: List[Dict], 
                            target_distribution: Dict[str, float]) -> List[Dict]:
        categorized = {'easy': [], 'medium': [], 'hard': []}
        for q in questions:
            diff = self.estimate_difficulty(q['question'], q.get('correct_answer', ''))
            categorized[diff].append(q)
        
        balanced = []
        total_questions = len(questions)
        for difficulty, ratio in target_distribution.items():
            count = int(total_questions * ratio)
            balanced.extend(categorized[difficulty][:count])
        
        return balanced

# ===================== PROMPT OPTIMIZER =====================

class PromptOptimizer:
    """Generate optimized prompts for tasks"""
    
    @staticmethod
    def get_question_generation_prompt(context: str, num_questions: int,
                                      difficulty: str, bloom_levels: List[str]) -> str:
        bloom_descriptions = {
            'remember': 'recall facts and basic concepts',
            'understand': 'explain ideas or concepts',
            'apply': 'use information in new situations',
            'analyze': 'draw connections among ideas',
            'evaluate': 'justify a decision or course of action',
            'create': 'produce new or original work'
        }
        
        bloom_instructions = '\n'.join([
            f"- {level.capitalize()}: {bloom_descriptions[level]}"
            for level in bloom_levels if level in bloom_descriptions
        ])
        
        prompt = f"""You are an expert educational content creator. Generate {num_questions} high-quality test questions based on the following study material.

Study Material:
{context}

Requirements:
- Difficulty Level: {difficulty}
- Cognitive Levels (Bloom's Taxonomy):
{bloom_instructions}

Return ONLY a valid JSON array as described."""
        return prompt
    
    @staticmethod
    def get_answer_evaluation_prompt(question: str, student_answer: str,
                                    reference_answer: str, context: str) -> str:
        prompt = f"""You are an expert educator evaluating a student's answer. Be fair, constructive, and encouraging.

Question: {question}

Student's Answer:
{student_answer}

Reference/Expected Answer:
{reference_answer}

Study Material Context:
{context}

Provide evaluation as JSON."""
        return prompt

# ===================== HUGGING FACE LLM INTEGRATION =====================

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class HuggingFaceLLM:
    """Wrapper for Hugging Face instruction-following LLM"""
    def __init__(self, model_name="mistral-7b-instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===================== CONCEPT EXTRACTION =====================

class ConceptExtractor:
    """Extract key concepts from notes for targeted testing"""
    
    def __init__(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            self.model = None
    
    def extract_key_concepts(self, text: str, top_n: int = 10) -> List[Dict]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        concepts = []
        for sentence in sentences:
            potential_concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
            concepts.extend(potential_concepts)
        
        from collections import Counter
        concept_counts = Counter(concepts)
        
        top_concepts = []
        for concept, count in concept_counts.most_common(top_n):
            example = next((s for s in sentences if concept in s), "")
            top_concepts.append({
                'concept': concept,
                'frequency': count,
                'example_context': example[:200]
            })
        
        return top_concepts
    
    def generate_concept_map(self, concepts: List[Dict]) -> Dict:
        if not self.model or not concepts:
            return {'nodes': [], 'edges': []}
        
        concept_map = {
            'nodes': [{'id': i, 'label': c['concept']} for i, c in enumerate(concepts)],
            'edges': []
        }
        
        try:
            embeddings = self.model.encode([c['concept'] for c in concepts])
            
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    similarity = cosine_similarity(
                        embeddings[i].reshape(1, -1),
                        embeddings[j].reshape(1, -1)
                    )[0][0]
                    
                    if similarity > 0.5:
                        concept_map['edges'].append({
                            'from': i,
                            'to': j,
                            'weight': float(similarity)
                        })
        except Exception as e:
            print(f"Concept map generation error: {e}")
        
        return concept_map
