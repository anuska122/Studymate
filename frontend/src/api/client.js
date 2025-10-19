const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

class StudyMateAPI {
  async request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`
    const config = {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    }
    
    try {
      const response = await fetch(url, config)
      if (!response.ok) {
        const error = await response.json().catch(() => ({ 
          detail: `HTTP ${response.status}` 
        }))
        throw new Error(error.detail || `Request failed: ${response.status}`)
      }
      return await response.json()
    } catch (error) {
      console.error('API Error:', error)
      throw error
    }
  }

  async checkHealth() {
    return this.request('/health')
  }

  async createNote(noteData) {
    return this.request('/api/notes', {
      method: 'POST',
      body: JSON.stringify({
        title: noteData.title,
        content: noteData.content,
        subject: noteData.subject,
        review_days: noteData.reviewDays || 7,
        use_semantic_chunking: noteData.useSemanticChunking !== false
      }),
    })
  }

  async generateTest(noteId, numQuestions = 5) {
    return this.request('/api/generate-test', {
      method: 'POST',
      body: JSON.stringify({
        note_id: noteId,
        num_questions: numQuestions,
        difficulty: 'medium',
        use_adaptive: false,
        bloom_levels: ['understand', 'apply', 'analyze']
      }),
    })
  }

  async evaluateTest(noteId, answers) {
    return this.request('/api/evaluate-test', {
      method: 'POST',
      body: JSON.stringify({
        note_id: noteId,
        answers: answers.map(a => ({
          question_id: a.questionId,
          answer: a.answer,
        })),
      }),
    })
  }

  async getNoteConcepts(noteId) {
    return this.request(`/api/notes/${noteId}/concepts`)
  }
}

export const api = new StudyMateAPI()
export default api