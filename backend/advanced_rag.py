
"""
Advanced RAG techniques for StudyMate
Implementing these to improve question generation and answer evaluation
"""

from typing import List, Dict, Tuple
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunker:
    """Split text into semantically meaningful chunks"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def chunk_by_similarity(self, text: str, threshold: float = 0.5, 
                           min_chunk_size: int = 100) -> List[str]:
        """
        Split text based on semantic similarity
        Sentences with low similarity start new chunks
        """
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
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

# QUERY EXPANSION
    
class QueryExpander:
    """Expand user queries to improve retrieval"""
    
    def expand_with_synonyms(self, query: str) -> List[str]:
        """Generate query variations using synonyms"""
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
        """Add subject context to query"""
        return f"{query} in the context of {note_subject}"

# HYBRID SEARCH (BM25 + Semantic)

class HybridRetriever:
    """Combine keyword-based and semantic search"""
    
    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model
    
    def retrieve(self, query: str, note_id: str, top_k: int = 5,
                 alpha: float = 0.5) -> List[Dict]:
        """
        Hybrid retrieval with alpha balancing
        alpha=1.0: pure semantic, alpha=0.0: pure keyword
        """
        try:
            # Semantic search
            semantic_results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k * 2, 10),
                where={"note_id": note_id}
            )
            
            # Keyword search
            keyword_results = self._keyword_search(query, note_id, min(top_k * 2, 10))
            
            # Combine and re-rank
            combined = self._fuse_results(semantic_results, keyword_results, alpha)
            
            return combined[:top_k]
        except Exception as e:
            print(f"Hybrid retrieval error: {e}")
            return []
    
    def _keyword_search(self, query: str, note_id: str, k: int) -> List[Dict]:
        """Simple keyword-based search"""
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
        """Reciprocal Rank Fusion"""
        scores = {}
        k = 60  # RRF parameter
        
        # Score semantic results
        if semantic.get('documents') and semantic['documents']:
            for rank, doc in enumerate(semantic['documents'][0]):
                scores[doc] = scores.get(doc, 0) + alpha / (k + rank + 1)
        
        # Score keyword results
        for rank, item in enumerate(keyword):
            doc = item['document']
            scores[doc] = scores.get(doc, 0) + (1 - alpha) / (k + rank + 1)
        
        # Sort by combined score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{'document': doc, 'score': score} for doc, score in ranked]

# SPACED REPETITION (SM-2 Algorithm)

class SpacedRepetition:
    """Implement Anki-style spaced repetition"""
    
    @staticmethod
    def calculate_next_interval(quality: int, interval: int, 
                               easiness_factor: float, repetitions: int) -> Tuple[int, float, int]:
        """
        SM-2 Algorithm
        
        Args:
            quality: 0-5 (how well did you answer?)
            interval: current interval in days
            easiness_factor: current easiness (default 2.5)
            repetitions: number of correct repetitions in a row
        
        Returns:
            (new_interval, new_easiness, new_repetitions)
        """
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
        """Convert test score to SM-2 quality rating"""
        if score >= 90:
            return 5  # Perfect
        elif score >= 80:
            return 4  # Good
        elif score >= 60:
            return 3  # Acceptable
        elif score >= 40:
            return 2  # Hard
        elif score >= 20:
            return 1  # Very hard
        else:
            return 0  # Failed

# QUESTION DIFFICULTY CALIBRATION


class QuestionDifficultyEstimator:
    """Estimate and calibrate question difficulty"""
    
    def estimate_difficulty(self, question: str, answer: str) -> str:
        """Estimate question difficulty based on characteristics"""
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
        """Balance question difficulties according to target distribution"""
        # Categorize questions
        categorized = {'easy': [], 'medium': [], 'hard': []}
        for q in questions:
            diff = self.estimate_difficulty(q['question'], q.get('correct_answer', ''))
            categorized[diff].append(q)
        
        # Select according to distribution
        balanced = []
        total_questions = len(questions)
        
        for difficulty, ratio in target_distribution.items():
            count = int(total_questions * ratio)
            balanced.extend(categorized[difficulty][:count])
        
        return balanced

# CONTEXT-AWARE PROMPTS

class PromptOptimizer:
    """Generate optimized prompts for different tasks"""
    
    @staticmethod
    def get_question_generation_prompt(context: str, num_questions: int,
                                      difficulty: str, bloom_levels: List[str]) -> str:
        """Generate optimized prompt for question generation"""
        
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

Question Types to Include:
1. Multiple Choice Questions (MCQ): Test understanding and application
2. Short Answer: Test recall and basic comprehension
3. Long Answer: Test analysis, evaluation, and synthesis

Guidelines:
- Ensure questions are clear, unambiguous, and directly related to the material
- Avoid trick questions or overly obscure details
- Include a mix of cognitive levels as specified
- For MCQs, ensure distractors are plausible but clearly incorrect
- Provide detailed explanations for correct answers

Return ONLY a valid JSON array with this structure:
[
  {{
    "question": "Clear, specific question text",
    "type": "mcq|short|long",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "Detailed correct answer",
    "explanation": "Why this is correct and how it relates to the material",
    "cognitive_level": "remember|understand|apply|analyze|evaluate|create",
    "difficulty": "easy|medium|hard"
  }}
]"""
        return prompt
    
    @staticmethod
    def get_answer_evaluation_prompt(question: str, student_answer: str,
                                    reference_answer: str, context: str) -> str:
        """Generate prompt for evaluating student answers"""
        
        prompt = f"""You are an expert educator evaluating a student's answer. Be fair, constructive, and encouraging.

Question: {question}

Student's Answer:
{student_answer}

Reference/Expected Answer:
{reference_answer}

Study Material Context:
{context}

Evaluation Criteria:
1. Accuracy: Is the answer factually correct?
2. Completeness: Does it cover all key points?
3. Understanding: Does it demonstrate comprehension?
4. Clarity: Is it well-articulated?

Provide evaluation as JSON:
{{
  "is_correct": true/false,
  "score": 0-100,
  "accuracy_score": 0-100,
  "completeness_score": 0-100,
  "understanding_score": 0-100,
  "feedback": "Constructive, encouraging feedback with specific strengths and areas for improvement",
  "key_points_covered": ["point1", "point2"],
  "key_points_missed": ["missing1", "missing2"],
  "improvement_suggestions": ["specific suggestion1", "specific suggestion2"],
  "exemplar_phrases": ["good phrase they used", "another good phrase"]
}}"""
        return prompt

# ACTIVE LEARNING - IDENTIFY WEAK AREAS

class WeaknessIdentifier:
    """Identify topics where student needs more practice"""
    
    def __init__(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            self.model = None
    
    def identify_weak_topics(self, test_history: List[Dict], 
                            threshold: float = 0.7) -> List[Dict]:
        """
        Analyze test history to find weak areas
        
        Args:
            test_history: List of {note_id, topic, questions, scores}
            threshold: Score below which topic is considered weak
        """
        topic_scores = {}
        
        for test in test_history:
            topic = test.get('topic', test.get('subject', 'unknown'))
            score = test.get('avg_score', test.get('score', 0))
            
            if topic not in topic_scores:
                topic_scores[topic] = []
            topic_scores[topic].append(score)
        
        weak_topics = []
        for topic, scores in topic_scores.items():
            avg_score = np.mean(scores)
            if avg_score < threshold * 100:
                weak_topics.append({
                    'topic': topic,
                    'avg_score': float(avg_score),
                    'attempts': len(scores),
                    'trend': 'improving' if self._is_improving(scores) else 'declining',
                    'priority': 'high' if avg_score < 50 else 'medium'
                })
        
        return sorted(weak_topics, key=lambda x: x['avg_score'])
    
    def _is_improving(self, scores: List[float]) -> bool:
        """Check if scores show improvement trend"""
        if len(scores) < 2:
            return False
        return scores[-1] > scores[0]
    
    def recommend_review_topics(self, weak_topics: List[Dict], 
                               all_notes: List[Dict]) -> List[Dict]:
        """Recommend specific notes to review based on weak topics"""
        recommendations = []
        
        for weak in weak_topics:
            matching_notes = [
                note for note in all_notes 
                if weak['topic'].lower() in note.get('subject', '').lower()
            ]
            
            if matching_notes:
                recommendations.append({
                    'weak_topic': weak['topic'],
                    'priority': weak['priority'],
                    'recommended_notes': matching_notes[:3],
                    'suggested_action': self._get_action_plan(weak)
                })
        
        return recommendations
    
    def _get_action_plan(self, weak_topic: Dict) -> str:
        """Generate personalized action plan"""
        score = weak_topic['avg_score']
        trend = weak_topic['trend']
        
        if score < 50:
            return f"Focus on fundamentals. Review basic concepts in {weak_topic['topic']}. Start with easier questions."
        elif score < 70:
            if trend == 'improving':
                return f"You're making progress! Continue practicing {weak_topic['topic']} with medium difficulty questions."
            else:
                return f"Review core concepts in {weak_topic['topic']}. Consider breaking it into smaller sub-topics."
        else:
            return f"Almost there! Practice application and analysis questions in {weak_topic['topic']}."

# ADAPTIVE TESTING
class AdaptiveTestGenerator:
    """Generate tests that adapt to student performance"""
    
    def generate_adaptive_test(self, note_id: str, student_history: List[Dict],
                              initial_difficulty: str = 'medium') -> Dict:
        """
        Generate a test that adapts based on student's historical performance
        """
        # Analyze student's performance on this topic
        topic_performance = self._analyze_topic_performance(note_id, student_history)
        
        # Determine appropriate difficulty distribution
        if topic_performance['mastery_level'] == 'beginner':
            difficulty_dist = {'easy': 0.6, 'medium': 0.3, 'hard': 0.1}
        elif topic_performance['mastery_level'] == 'intermediate':
            difficulty_dist = {'easy': 0.2, 'medium': 0.5, 'hard': 0.3}
        else:  # advanced
            difficulty_dist = {'easy': 0.1, 'medium': 0.3, 'hard': 0.6}
        
        # Determine cognitive level distribution
        if topic_performance['mastery_level'] == 'beginner':
            bloom_levels = ['remember', 'understand', 'apply']
        elif topic_performance['mastery_level'] == 'intermediate':
            bloom_levels = ['understand', 'apply', 'analyze']
        else:
            bloom_levels = ['apply', 'analyze', 'evaluate', 'create']
        
        return {
            'note_id': note_id,
            'difficulty_distribution': difficulty_dist,
            'bloom_levels': bloom_levels,
            'recommended_questions': 5,
            'focus_areas': topic_performance.get('weak_subtopics', []),
            'mastery_level': topic_performance['mastery_level']
        }
    
    def _analyze_topic_performance(self, note_id: str, 
                                   history: List[Dict]) -> Dict:
        """Analyze student's performance on a specific topic"""
        topic_tests = [t for t in history if t.get('note_id') == note_id]
        
        if not topic_tests:
            return {'mastery_level': 'beginner', 'weak_subtopics': []}
        
        avg_score = np.mean([t.get('score', 0) for t in topic_tests])
        
        if avg_score >= 85:
            mastery = 'advanced'
        elif avg_score >= 70:
            mastery = 'intermediate'
        else:
            mastery = 'beginner'
        
        return {
            'mastery_level': mastery,
            'avg_score': float(avg_score),
            'attempts': len(topic_tests),
            'weak_subtopics': []
        }

# PERFORMANCE ANALYTICS
class PerformanceAnalyzer:
    """Analyze and visualize student performance"""
    
    def generate_analytics(self, test_history: List[Dict]) -> Dict:
        """Generate comprehensive analytics"""
        if not test_history:
            return {}
        
        scores = [t.get('score', 0) for t in test_history]
        
        analytics = {
            'overall_stats': {
                'total_tests': len(test_history),
                'avg_score': float(np.mean(scores)),
                'median_score': float(np.median(scores)),
                'std_dev': float(np.std(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'improvement_rate': self._calculate_improvement_rate(scores)
            },
            'by_subject': self._analyze_by_subject(test_history),
            'learning_curve': self._generate_learning_curve(test_history),
            'predictions': self._predict_mastery_date(test_history)
        }
        
        return analytics
    
    def _calculate_improvement_rate(self, scores: List[float]) -> float:
        """Calculate the rate of improvement"""
        if len(scores) < 2:
            return 0.0
        
        # Simple linear regression
        x = np.arange(len(scores))
        slope, _ = np.polyfit(x, scores, 1)
        return float(slope)
    
    def _analyze_by_subject(self, history: List[Dict]) -> Dict:
        """Break down performance by subject"""
        subjects = {}
        for test in history:
            subject = test.get('subject', test.get('topic', 'unknown'))
            if subject not in subjects:
                subjects[subject] = []
            subjects[subject].append(test.get('score', 0))
        
        return {
            subject: {
                'avg_score': float(np.mean(scores)),
                'attempts': len(scores),
                'trend': 'improving' if len(scores) >= 2 and scores[-1] > scores[0] else 'stable'
            }
            for subject, scores in subjects.items()
        }
    
    def _generate_learning_curve(self, history: List[Dict]) -> List[Dict]:
        """Generate data for learning curve visualization"""
        return [
            {
                'attempt': i + 1,
                'score': test.get('score', 0),
                'date': test.get('date', ''),
                'subject': test.get('subject', test.get('topic', 'unknown'))
            }
            for i, test in enumerate(history)
        ]
    
    def _predict_mastery_date(self, history: List[Dict]) -> Dict:
        """Predict when student will achieve mastery (85%+)"""
        if len(history) < 3:
            return {'prediction': 'Need more data', 'tests_until_mastery': None}
        
        scores = [t.get('score', 0) for t in history]
        improvement_rate = self._calculate_improvement_rate(scores)
        
        if improvement_rate <= 0:
            return {
                'prediction': 'Recommend reviewing fundamentals',
                'tests_until_mastery': None,
                'current_trajectory': 'negative'
            }
        
        current_score = scores[-1]
        tests_needed = max(0, int((85 - current_score) / improvement_rate))
        
        return {
            'tests_until_mastery': tests_needed,
            'current_trajectory': 'positive' if improvement_rate > 0 else 'negative',
            'confidence': 'medium',
            'estimated_score_next_test': min(100, current_score + improvement_rate)
        }

# CONCEPT EXTRACTION

class ConceptExtractor:
    """Extract key concepts from notes for targeted testing"""
    
    def __init__(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            self.model = None
    
    def extract_key_concepts(self, text: str, top_n: int = 10) -> List[Dict]:
        """
        Extract key concepts using TF-IDF-like approach with embeddings
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Extract noun phrases (simple approach)
        concepts = []
        for sentence in sentences:
            # Extract capitalized words and phrases (potential concepts)
            potential_concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
            concepts.extend(potential_concepts)
        
        # Count frequency
        from collections import Counter
        concept_counts = Counter(concepts)
        
        # Return top concepts with context
        top_concepts = []
        for concept, count in concept_counts.most_common(top_n):
            # Find example sentence
            example = next((s for s in sentences if concept in s), "")
            top_concepts.append({
                'concept': concept,
                'frequency': count,
                'example_context': example[:200]
            })
        
        return top_concepts
    
    def generate_concept_map(self, concepts: List[Dict]) -> Dict:
        """Generate a concept map showing relationships"""
        if not self.model or not concepts:
            return {'nodes': [], 'edges': []}
        
        # Simple concept map structure
        concept_map = {
            'nodes': [{'id': i, 'label': c['concept']} for i, c in enumerate(concepts)],
            'edges': []
        }
        
        try:
            # Find relationships between concepts (co-occurrence in sentences)
            embeddings = self.model.encode([c['concept'] for c in concepts])
            
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    similarity = cosine_similarity(
                        embeddings[i].reshape(1, -1),
                        embeddings[j].reshape(1, -1)
                    )[0][0]
                    
                    if similarity > 0.5:  # Threshold for relationship
                        concept_map['edges'].append({
                            'from': i,
                            'to': j,
                            'weight': float(similarity)
                        })
        except Exception as e:
            print(f"Concept map generation error: {e}")
        
        return concept_map