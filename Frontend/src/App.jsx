import { useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

function App() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [results, setResults] = useState([])
  const [activeTab, setActiveTab] = useState('answer')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function submitQuestion(nextQuestion = question) {
    const trimmedQuestion = nextQuestion.trim()
    if (!trimmedQuestion || loading) return

    setQuestion(trimmedQuestion)
    setLoading(true)
    setError('')
    setActiveTab('answer')
    try {
      const response = await fetch(`${API_URL}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmedQuestion, top_k: 5, candidate_k: 20 }),
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || 'The document service could not answer.')
      setAnswer(payload.answer)
      setResults(payload.results || [])
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoading(false)
    }
  }

  function handleSubmit(event) {
    event.preventDefault()
    submitQuestion()
  }

  return (
    <div className="app-shell">
      <header className="top-header">
        <div className="logo-block">
          <div className="logo-badge" aria-hidden="true" />
          <div className="logo-wordmark">SNOW</div>
        </div>

        <nav className="top-nav" aria-label="Main navigation">
          <a href="#">Solutions</a>
          <a href="#">Services</a>
          <a href="#">Resources</a>
          <a href="#">Company</a>
        </nav>
      </header>

      <main className="single-page-shell">
        <section className="chat-panel">
          <div className="motion-stage" aria-hidden="true">
            <div className="document-sheet sheet-back"><span>DOC / 03</span><i /></div>
            <div className="document-sheet sheet-middle"><span>RETRIEVAL</span><i /><i /></div>
            <div className="document-sheet sheet-front"><span>ASK</span><strong>?</strong></div>
          </div>
          <h2 className="chat-title">Ask your documents.</h2>

          <div className="query-wrap">
            <form className="query-form" onSubmit={handleSubmit}>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="What would you like to understand?"
                rows="2"
                aria-label="Question"
              />
              <button type="submit" className="submit-button" disabled={loading || !question.trim()}>
                {loading ? 'Searching...' : 'Find answer'} <span>↗</span>
              </button>
            </form>
          </div>

          {error && (
            <div className="error-banner">
              <strong>Could not complete that request.</strong>
              <span>{error}</span>
            </div>
          )}

          <section className="results-section">
            <div className="results-header">
              <h3>{answer ? 'A grounded answer' : 'Your answer will land here'}</h3>
            </div>

            <div className="result-panel">
              <div className="tabs">
                <button className={activeTab === 'answer' ? 'tab active' : 'tab'} onClick={() => setActiveTab('answer')}>
                  Answer
                </button>
                <button className={activeTab === 'sources' ? 'tab active' : 'tab'} onClick={() => setActiveTab('sources')}>
                  Sources <span>{results.length || ''}</span>
                </button>
              </div>

              {loading ? (
                <div className="loading-state">
                  <div className="loader" />
                  <p>Reading the most relevant passages...</p>
                </div>
              ) : activeTab === 'answer' ? (
                <AnswerView answer={answer} />
              ) : (
                <SourcesView results={results} />
              )}
            </div>
          </section>
        </section>
      </main>
    </div>
  )
}

function AnswerView({ answer }) {
  if (!answer) {
    return (
      <div className="empty-answer" aria-live="polite" />
    )
  }

  return (
    <article className="answer-view">
      <div className="answer-label">SYNTHESIZED FROM YOUR DOCUMENTS</div>
      <div className="answer-copy">
        {answer.split('\n').map((line, index) => (
          <p key={`${line}-${index}`}>{line || '\u00a0'}</p>
        ))}
      </div>
      <div className="answer-note">
        <span>◈</span> Answer generated from retrieved passages and reranked for relevance.
      </div>
    </article>
  )
}

function SourcesView({ results }) {
  if (!results.length) {
    return (
      <div className="empty-answer">
        <div className="empty-number">S</div>
        <h3>No sources yet</h3>
        <p>Ask a question to see the ranked passages used by the answer.</p>
      </div>
    )
  }

  return (
    <div className="sources-view">
      {results.map((result, index) => (
        <article className="source-card" key={`${result.metadata?.source}-${result.metadata?.page}-${index}`}>
          <div className="source-rank">0{index + 1}</div>
          <div className="source-body">
            <div className="source-meta">
              <span>{result.metadata?.source || 'Unknown document'}</span>
              <span>PAGE {result.metadata?.page || '—'}</span>
            </div>
            <p>{result.text}</p>
          </div>
          <div className="source-score">{result.rerank_score?.toFixed(2)}</div>
        </article>
      ))}
    </div>
  )
}

export default App
