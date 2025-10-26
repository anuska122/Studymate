import { useState, useEffect } from 'react'
import { BookOpen, Brain, Calendar, Plus, Clock, CheckCircle, AlertCircle, Loader, TrendingUp } from 'lucide-react'
import { api } from './api/client'

function App() {
  const [notes, setNotes] = useState([])
  const [selectedNote, setSelectedNote] = useState(null)
  const [isCreating, setIsCreating] = useState(false)
  const [newNote, setNewNote] = useState({
    title: '',
    content: '',
    subject: '',
    reviewDays: 7,
    useSemanticChunking: true
  })
  
  const [testMode, setTestMode] = useState(false)
  const [currentTest, setCurrentTest] = useState(null)
  const [userAnswers, setUserAnswers] = useState({})
  const [testResults, setTestResults] = useState(null)
  
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [backendStatus, setBackendStatus] = useState('checking')
  const [showConcepts, setShowConcepts] = useState(false)
  const [concepts, setConcepts] = useState(null)

  useEffect(() => {
    checkBackendHealth()
    const interval = setInterval(checkBackendHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  const checkBackendHealth = async () => {
    try {
      const health = await api.checkHealth()
      setBackendStatus(health.ollama === 'connected' ? 'connected' : 'disconnected')
      if (health.ollama === 'disconnected') {
        setError('Ollama not running. Start: ollama serve')
      } else {
        setError(null)
      }
    } catch (err) {
      setBackendStatus('disconnected')
      setError('Backend offline. Run: python backend/main.py')
    }
  }

  const handleCreateNote = async () => {
    if (!newNote.title || !newNote.content) {
      setError('Please fill in title and content')
      return
    }

    if (newNote.content.split(' ').length < 20) {
      setError('Note too short. Add at least 20 words.')
      return
    }

    setIsLoading(true)
    setError(null)
    
    try {
      const createdNote = await api.createNote(newNote)
      setNotes([...notes, createdNote])
      setNewNote({ title: '', content: '', subject: '', reviewDays: 7, useSemanticChunking: true })
      setIsCreating(false)
      alert(`✅ Note created with ${createdNote.chunks_created} chunks!`)
    } catch (err) {
      setError(`Failed: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  const startTest = async (noteId) => {
    setIsLoading(true)
    setError(null)
    
    try {
      const test = await api.generateTest(noteId, 5)
      if (!test.questions || test.questions.length === 0) {
        throw new Error('No questions generated')
      }
      setCurrentTest(test)
      setTestMode(true)
      setUserAnswers({})
      setTestResults(null)
    } catch (err) {
      setError(`Test failed: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  const submitTest = async () => {
    const answered = Object.keys(userAnswers).length
    const total = currentTest.questions.length
    
    if (answered !== total) {
      setError(`Answer all questions (${answered}/${total})`)
      return
    }

    setIsLoading(true)
    setError(null)
    
    try {
      const answers = Object.entries(userAnswers).map(([qId, answer]) => ({
        questionId: parseInt(qId),
        answer: answer,
      }))
      
      const results = await api.evaluateTest(currentTest.note_id, answers)
      setTestResults(results)
    } catch (err) {
      setError(`Evaluation failed: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  const loadConcepts = async (noteId) => {
    try {
      setIsLoading(true)
      const conceptData = await api.getNoteConcepts(noteId)
      setConcepts(conceptData)
      setShowConcepts(true)
    } catch (err) {
      setError('Concepts failed: ' + err.message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="h-screen flex flex-col bg-gray-100">
      <header className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-3 sm:p-4 shadow-lg">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-2 sm:space-x-3">
            <Brain size={24} className="sm:w-8 sm:h-8" />
            <div>
              <h1 className="text-lg sm:text-2xl font-bold">StudyMate</h1>
              <p className="text-xs text-blue-200 hidden sm:block">AI-Powered Learning</p>
            </div>
          </div>
          <div className={`flex items-center space-x-1 sm:space-x-2 text-xs sm:text-sm px-2 sm:px-4 py-1 sm:py-2 rounded-lg ${
            backendStatus === 'connected' ? 'bg-green-500' : 
            backendStatus === 'disconnected' ? 'bg-red-500' : 'bg-yellow-500'
          }`}>
            <div className={`w-2 h-2 rounded-full bg-white ${
              backendStatus === 'connected' ? 'animate-pulse' : ''
            }`}></div>
            <span className="font-medium hidden sm:inline">
              {backendStatus === 'connected' ? 'Connected' : 
               backendStatus === 'disconnected' ? 'Offline' : 'Checking...'}
            </span>
            <span className="font-medium sm:hidden">
              {backendStatus === 'connected' ? '✓' : 
               backendStatus === 'disconnected' ? '✗' : '...'}
            </span>
          </div>
        </div>
      </header>

      {error && backendStatus === 'disconnected' && (
        <div className="bg-red-500 text-white px-4 py-2 text-center text-xs sm:text-sm">
          ⚠️ {error}
        </div>
      )}

      <main className="flex-1 overflow-hidden">
        {testMode ? (
          <TestView 
            currentTest={currentTest}
            userAnswers={userAnswers}
            setUserAnswers={setUserAnswers}
            testResults={testResults}
            isLoading={isLoading}
            error={error}
            submitTest={submitTest}
            onBack={() => {
              setTestMode(false)
              setTestResults(null)
              setCurrentTest(null)
              setUserAnswers({})
              setError(null)
            }}
          />
        ) : (
          <NotesView 
            notes={notes}
            selectedNote={selectedNote}
            setSelectedNote={setSelectedNote}
            isCreating={isCreating}
            setIsCreating={setIsCreating}
            newNote={newNote}
            setNewNote={setNewNote}
            error={error}
            setError={setError}
            isLoading={isLoading}
            backendStatus={backendStatus}
            handleCreateNote={handleCreateNote}
            startTest={startTest}
            loadConcepts={loadConcepts}
            showConcepts={showConcepts}
            setShowConcepts={setShowConcepts}
            concepts={concepts}
          />
        )}
      </main>

      <footer className="bg-gray-800 text-gray-400 text-center py-2 text-xs hidden sm:block">
        Powered by Llama 3.1 + ChromaDB + React + Vite
      </footer>
    </div>
  )
}

function NotesView({ 
  notes, selectedNote, setSelectedNote, isCreating, setIsCreating,
  newNote, setNewNote, error, setError, isLoading, backendStatus,
  handleCreateNote, startTest, loadConcepts, showConcepts, setShowConcepts, concepts
}) {
  const [showMobileMenu, setShowMobileMenu] = useState(false)
  
  return (
    <div className="flex h-full relative">
      <button 
        onClick={() => setShowMobileMenu(!showMobileMenu)}
        className="lg:hidden fixed bottom-4 right-4 z-20 p-4 bg-blue-500 text-white rounded-full shadow-lg hover:bg-blue-600"
      >
        <BookOpen size={24} />
      </button>

      <aside className={`
        w-full lg:w-1/3 border-r border-gray-200 overflow-y-auto bg-gray-50
        fixed lg:relative inset-0 z-10 lg:z-0
        transition-transform duration-300 ease-in-out
        ${showMobileMenu ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        <div className="p-3 sm:p-4 border-b border-gray-200 flex justify-between items-center bg-white sticky top-0 z-10">
          <h2 className="text-base sm:text-lg font-semibold text-gray-800">My Notes</h2>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => {
                setIsCreating(true)
                setShowMobileMenu(false)
              }}
              disabled={isLoading || backendStatus !== 'connected'}
              className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors disabled:bg-gray-400"
            >
              <Plus size={18} className="sm:w-5 sm:h-5" />
            </button>
            <button 
              onClick={() => setShowMobileMenu(false)}
              className="lg:hidden p-2 text-gray-600 hover:bg-gray-100 rounded-lg"
            >
              ✕
            </button>
          </div>
        </div>
        <div className="p-2 sm:p-3">
          {notes.length === 0 ? (
            <div className="text-center p-6 sm:p-8 text-gray-400">
              <BookOpen size={40} className="sm:w-12 sm:h-12 mx-auto mb-3 opacity-50" />
              <p className="text-sm">No notes yet</p>
              <p className="text-xs mt-2">Create your first note!</p>
            </div>
          ) : (
            notes.map(note => (
              <div
                key={note.id}
                onClick={() => { 
                  setSelectedNote(note)
                  setShowConcepts(false)
                  setShowMobileMenu(false)
                }}
                className={`p-3 sm:p-4 mb-2 rounded-lg cursor-pointer transition-all ${
                  selectedNote?.id === note.id 
                    ? 'bg-blue-50 border-2 border-blue-300 shadow-md' 
                    : 'bg-white border border-gray-200 hover:shadow-md'
                }`}
              >
                <h3 className="font-semibold text-sm sm:text-base text-gray-900">{note.title}</h3>
                <p className="text-xs sm:text-sm text-blue-600 mt-1 font-medium">{note.subject}</p>
                <div className="flex items-center justify-between mt-2 sm:mt-3">
                  <div className="flex items-center text-xs text-gray-500">
                    <Calendar size={12} className="mr-1" />
                    <span className="hidden sm:inline">{note.next_review}</span>
                    <span className="sm:hidden">{note.next_review.split(' ')[0]}</span>
                  </div>
                  {note.chunks_created && (
                    <div className="text-xs text-gray-400">
                      {note.chunks_created} chunks
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </aside>

      <section className="flex-1 overflow-y-auto bg-white">
        {isCreating ? (
          <CreateNoteForm 
            newNote={newNote}
            setNewNote={setNewNote}
            error={error}
            isLoading={isLoading}
            handleCreateNote={handleCreateNote}
            onCancel={() => { setIsCreating(false); setError(null); }}
          />
        ) : selectedNote ? (
          <NoteDetail 
            selectedNote={selectedNote}
            error={error}
            isLoading={isLoading}
            backendStatus={backendStatus}
            startTest={startTest}
            loadConcepts={loadConcepts}
            showConcepts={showConcepts}
            concepts={concepts}
          />
        ) : (
          <div className="flex items-center justify-center h-full text-gray-400 p-4">
            <div className="text-center">
              <BookOpen size={48} className="sm:w-16 sm:h-16 mx-auto mb-4 opacity-50" />
              <p className="text-base sm:text-lg">Select a note or create new</p>
            </div>
          </div>
        )}
      </section>
    </div>
  )
}

function CreateNoteForm({ newNote, setNewNote, error, isLoading, handleCreateNote, onCancel }) {
  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-4xl mx-auto">
      <h2 className="text-2xl sm:text-3xl font-bold mb-4 sm:mb-6 text-gray-800">Create New Note</h2>
      
      {error && (
        <div className="mb-4 p-3 sm:p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded flex items-center text-sm">
          <AlertCircle size={18} className="mr-2 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Title *</label>
          <input
            type="text"
            placeholder="e.g., Introduction to Biology"
            value={newNote.title}
            onChange={(e) => setNewNote({ ...newNote, title: e.target.value })}
            className="w-full p-2 sm:p-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none text-sm sm:text-base"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Subject *</label>
          <input
            type="text"
            placeholder="e.g., Biology, Physics"
            value={newNote.subject}
            onChange={(e) => setNewNote({ ...newNote, subject: e.target.value })}
            className="w-full p-2 sm:p-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none text-sm sm:text-base"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Content * <span className="text-gray-500 text-xs">(min 20 words)</span>
          </label>
          <textarea
            placeholder="Write detailed notes..."
            value={newNote.content}
            onChange={(e) => setNewNote({ ...newNote, content: e.target.value })}
            className="w-full p-2 sm:p-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none h-48 sm:h-64 resize-none text-sm sm:text-base"
          />
          <div className="text-xs text-gray-500 mt-1">
            Words: {newNote.content.split(' ').filter(w => w).length}
          </div>
        </div>
        
        <div className="flex flex-col sm:flex-row sm:items-center space-y-3 sm:space-y-0 sm:space-x-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Review after:</label>
            <select
              value={newNote.reviewDays}
              onChange={(e) => setNewNote({ ...newNote, reviewDays: parseInt(e.target.value) })}
              className="p-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 text-sm w-full sm:w-auto"
            >
              <option value={1}>1 day</option>
              <option value={3}>3 days</option>
              <option value={7}>7 days</option>
              <option value={14}>14 days</option>
              <option value={30}>30 days</option>
            </select>
          </div>
          
          <div className="flex items-center">
            <input
              type="checkbox"
              id="semantic"
              checked={newNote.useSemanticChunking}
              onChange={(e) => setNewNote({ ...newNote, useSemanticChunking: e.target.checked })}
              className="mr-2"
            />
            <label htmlFor="semantic" className="text-sm text-gray-700">
              Semantic chunking
            </label>
          </div>
        </div>
        
        <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-3 pt-4">
          <button
            onClick={handleCreateNote}
            disabled={isLoading}
            className="w-full sm:w-auto px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 font-medium disabled:bg-gray-400 flex items-center justify-center space-x-2"
          >
            {isLoading && <Loader size={16} className="animate-spin" />}
            <span>{isLoading ? 'Creating...' : 'Save Note'}</span>
          </button>
          <button
            onClick={onCancel}
            className="w-full sm:w-auto px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 font-medium"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}

function NoteDetail({ selectedNote, error, isLoading, backendStatus, startTest, loadConcepts, showConcepts, concepts }) {
  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-4xl mx-auto">
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start mb-4 sm:mb-6 space-y-3 sm:space-y-0">
        <div>
          <h2 className="text-2xl sm:text-3xl font-bold text-gray-800 mb-2">{selectedNote.title}</h2>
          <span className="text-sm sm:text-base text-blue-600 font-medium">{selectedNote.subject}</span>
        </div>
        <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-2">
          <button
            onClick={() => loadConcepts(selectedNote.id)}
            disabled={isLoading}
            className="px-3 sm:px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 font-medium flex items-center justify-center space-x-2 disabled:bg-gray-400 text-sm"
          >
            <TrendingUp size={16} className="sm:w-[18px] sm:h-[18px]" />
            <span>Concepts</span>
          </button>
          <button
            onClick={() => startTest(selectedNote.id)}
            disabled={isLoading || backendStatus !== 'connected'}
            className="px-4 sm:px-6 py-2 sm:py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 font-medium flex items-center justify-center space-x-2 disabled:bg-gray-400 text-sm"
          >
            {isLoading ? <Loader size={18} className="sm:w-5 sm:h-5 animate-spin" /> : <Brain size={18} className="sm:w-5 sm:h-5" />}
            <span>{isLoading ? 'Generating...' : 'Start Test'}</span>
          </button>
        </div>
      </div>
      
      {error && (
        <div className="mb-4 p-3 sm:p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded flex items-center text-sm">
          <AlertCircle size={18} className="mr-2 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
      
      {showConcepts && concepts && (
        <div className="mb-4 sm:mb-6 bg-purple-50 p-4 sm:p-6 rounded-lg border-2 border-purple-200">
          <h3 className="text-lg sm:text-xl font-bold text-purple-900 mb-3 sm:mb-4">Key Concepts</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
            {concepts.key_concepts.map((concept, idx) => (
              <div key={idx} className="bg-white p-3 sm:p-4 rounded-lg border border-purple-200">
                <div className="font-semibold text-sm sm:text-base text-purple-700">{concept.concept}</div>
                <div className="text-xs text-gray-500 mt-1">Frequency: {concept.frequency}</div>
                <div className="text-xs text-gray-600 mt-2 line-clamp-2">{concept.example_context.substring(0, 80)}...</div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      <div className="bg-gray-50 p-4 sm:p-6 rounded-lg border border-gray-200">
        <h3 className="text-xs sm:text-sm font-semibold text-gray-600 mb-2 sm:mb-3 uppercase">Content</h3>
        <p className="text-sm sm:text-base text-gray-700 leading-relaxed whitespace-pre-wrap">{selectedNote.content}</p>
      </div>
      
      <div className="mt-4 sm:mt-6 flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-2 sm:space-y-0 text-xs sm:text-sm text-gray-500">
        <div className="flex items-center">
          <Clock size={14} className="sm:w-4 sm:h-4 mr-2" />
          <span>Created: {selectedNote.date}</span>
        </div>
        <div className="flex items-center">
          <Calendar size={14} className="sm:w-4 sm:h-4 mr-2" />
          <span>Review: {selectedNote.next_review}</span>
        </div>
      </div>
    </div>
  )
}

function TestView({ currentTest, userAnswers, setUserAnswers, testResults, isLoading, error, submitTest, onBack }) {
  if (testResults) {
    return <TestResults testResults={testResults} onBack={onBack} />
  }

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-4xl mx-auto">
      <div className="mb-6 sm:mb-8">
        <h2 className="text-2xl sm:text-3xl font-bold text-gray-800 mb-2">Test</h2>
        <p className="text-sm sm:text-base text-gray-600">Answer all questions</p>
        <span className="inline-block mt-3 px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs sm:text-sm">
          {currentTest?.questions?.length || 0} Questions
        </span>
      </div>

      {error && (
        <div className="mb-4 p-3 sm:p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded flex items-center text-sm">
          <AlertCircle size={18} className="mr-2 flex-shrink-0" />
          {error}
        </div>
      )}

      <div className="space-y-4 sm:space-y-6">
        {currentTest?.questions?.map((q, idx) => (
          <div key={idx} className="bg-white p-4 sm:p-6 rounded-lg border-2 border-gray-200 shadow-sm">
            <h3 className="font-semibold text-base sm:text-lg text-gray-800 mb-3 sm:mb-4">
              Q{idx + 1}: {q.question}
            </h3>
            
            {q.type === 'mcq' && q.options ? (
              <div className="space-y-2">
                {q.options.map((option, i) => (
                  <label key={i} className="flex items-start p-3 border border-gray-300 rounded-lg cursor-pointer hover:bg-blue-50 transition-colors">
                    <input
                      type="radio"
                      name={`q${idx}`}
                      value={option}
                      checked={userAnswers[idx] === option}
                      onChange={(e) => setUserAnswers({...userAnswers, [idx]: e.target.value})}
                      className="mr-3 mt-1 flex-shrink-0"
                    />
                    <span className="text-sm sm:text-base">{option}</span>
                  </label>
                ))}
              </div>
            ) : (
              <textarea
                placeholder="Type your answer..."
                value={userAnswers[idx] || ''}
                onChange={(e) => setUserAnswers({...userAnswers, [idx]: e.target.value})}
                className="w-full p-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none h-20 sm:h-24 text-sm sm:text-base"
              />
            )}
          </div>
        ))}
      </div>

      <div className="mt-6 sm:mt-8 flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-4">
        <button
          onClick={submitTest}
          disabled={isLoading}
          className="w-full sm:w-auto px-8 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 font-medium disabled:bg-gray-400 flex items-center justify-center space-x-2"
        >
          {isLoading && <Loader size={16} className="animate-spin" />}
          <span>{isLoading ? 'Evaluating...' : 'Submit'}</span>
        </button>
        <button
          onClick={onBack}
          disabled={isLoading}
          className="w-full sm:w-auto px-8 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 font-medium"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}

function TestResults({ testResults, onBack }) {
  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-4xl mx-auto">
      <div className="text-center mb-6 sm:mb-8">
        <h2 className="text-2xl sm:text-3xl font-bold text-gray-800 mb-4">Results</h2>
        <div className="inline-block bg-gradient-to-br from-blue-100 to-blue-200 rounded-2xl px-8 sm:px-12 py-5 sm:py-6 shadow-lg">
          <div className="text-5xl sm:text-6xl font-bold text-blue-700">
            {Math.round(testResults.overall_score)}%
          </div>
        </div>
        
        {testResults.spaced_repetition && (
          <div className="mt-4 sm:mt-6 p-3 sm:p-4 bg-green-50 border border-green-200 rounded-lg inline-block">
            <div className="flex items-center text-green-800 text-sm sm:text-base">
              <Calendar size={16} className="sm:w-[18px] sm:h-[18px] mr-2" />
              <span className="font-medium">
                Next Review: {testResults.spaced_repetition.next_review_date}
              </span>
            </div>
          </div>
        )}
      </div>

      <div className="space-y-3 sm:space-y-4">
        {testResults.results.map((result, idx) => (
          <div key={idx} className={`p-4 sm:p-6 rounded-lg border-2 ${
            result.is_correct ? 'bg-green-50 border-green-300' : 'bg-yellow-50 border-yellow-300'
          }`}>
            <div className="flex items-start">
              {result.is_correct ? (
                <CheckCircle className="text-green-600 mr-3 mt-1 flex-shrink-0" size={20} />
              ) : (
                <AlertCircle className="text-yellow-600 mr-3 mt-1 flex-shrink-0" size={20} />
              )}
              <div className="flex-1 min-w-0">
                <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start mb-2">
                  <h3 className="font-semibold text-sm sm:text-base">Q{idx + 1}</h3>
                  <span className="text-lg sm:text-xl font-bold mt-1 sm:mt-0">{result.score}%</span>
                </div>
                <p className="text-sm sm:text-base text-gray-700 break-words">{result.feedback}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <button
        onClick={onBack}
        className="mt-6 sm:mt-8 px-8 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 font-medium w-full"
      >
        Back to Notes
      </button>
    </div>
  )
}

export default App