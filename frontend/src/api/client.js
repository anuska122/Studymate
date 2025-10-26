// API client for StudyMate frontend

const API_BASE_URL = "http://localhost:8000";

class ApiClient {
  async request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      if (error.message === "Failed to fetch") {
        throw new Error("Cannot connect to backend. Make sure it's running on port 8000");
      }
      throw error;
    }
  }

  async checkHealth() {
    try {
      const response = await this.request("/health");
      // it maps Ollama status to what frontend expects
      return {
        ...response,
        ollama: response.ollama || "connected"
      };
    } catch (error) {
      console.error("Health check failed:", error);
      throw error;
    }
  }

  async createNote(noteData) {
    return this.request("/api/notes", {
      method: "POST",
      body: JSON.stringify({
        title: noteData.title,
        content: noteData.content,
        subject: noteData.subject,
        review_days: noteData.reviewDays,
        use_semantic_chunking: noteData.useSemanticChunking,
      }),
    });
  }

  async getNoteConcepts(noteId) {
    return this.request(`/api/notes/${noteId}/concepts`);
  }

  async generateTest(noteId, numQuestions = 5, difficulty = "medium") {
    return this.request("/api/generate-test", {
      method: "POST",
      body: JSON.stringify({
        note_id: noteId,
        num_questions: numQuestions,
        difficulty: difficulty,
      }),
    });
  }

  async evaluateTest(noteId, answers) {
    return this.request("/api/evaluate-test", {
      method: "POST",
      body: JSON.stringify({
        note_id: noteId,
        answers: answers.map(ans => ({
          question_id: ans.questionId,
          answer: ans.answer
        })),
      }),
    });
  }
}

export const api = new ApiClient();