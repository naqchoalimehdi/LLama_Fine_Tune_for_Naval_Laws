
import { useState } from 'react'
import './App.css'

function App() {
  const [messages, setMessages] = useState([
    { role: 'system', content: 'Welcome. I am the Royal Saudi Navy Legal Expert. How can I assist you with maritime law today?' }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [training, setTraining] = useState(false)
  const [status, setStatus] = useState('')

  // Model Parameters
  const [temperature, setTemperature] = useState(0.3)
  const [maxTokens, setMaxTokens] = useState(250)
  const [topP, setTopP] = useState(0.8)
  const [repPenalty, setRepPenalty] = useState(1.2)

  const handleUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    setStatus('Uploading and processing PDF...')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('http://localhost:8000/upload_pdf', {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      if (res.ok) {
        setStatus(`✅ ${data.message}`)
      } else {
        setStatus(`❌ Error: ${data.detail}`)
      }
    } catch (err) {
      setStatus(`❌ Connection Error: ${err.message}`)
    }
  }

  const handleTrain = async () => {
    setTraining(true)
    setStatus('Training model... This usually takes 30-60 mins.')
    try {
      const res = await fetch('http://localhost:8000/train', { method: 'POST' })
      const data = await res.json()
      if (res.ok) {
        setStatus('✅ Training completed successfully! Model saved to Drive.')
      } else {
        setStatus(`❌ Training Error: ${data.detail}`)
      }
    } catch (err) {
      setStatus(`❌ Connection Error: ${err.message}`)
    }
    setTraining(false)
  }

  const handleSend = async () => {
    if (!input.trim()) return
    const userMsg = { role: 'user', content: input }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMsg.content,
          temperature: parseFloat(temperature),
          max_new_tokens: parseInt(maxTokens),
          top_p: parseFloat(topP),
          repetition_penalty: parseFloat(repPenalty)
        })
      })
      const data = await res.json()
      if (res.ok) {
        setMessages(prev => [...prev, { role: 'assistant', content: data.response }])
      } else {
        setMessages(prev => [...prev, { role: 'error', content: `Error: ${data.detail}` }])
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'error', content: "Failed to connect to backend." }])
    }
    setLoading(false)
  }

  return (
    <div className="app-container">
      <header className="navbar">
        <div className="logo">⚓ Royal Saudi Navy Legal Expert</div>
      </header>

      <main className="main-content">
        <div className="control-panel">
          <div className="card">
            <h3>Documents</h3>
            <div className="file-input-wrapper">
              <input type="file" accept=".pdf" onChange={handleUpload} id="file-upload" />
              <label htmlFor="file-upload" className="btn-secondary">Upload PDF</label>
            </div>
          </div>

          <div className="card">
            <h3>Training</h3>
            <p className="status-text">{status || "System Ready"}</p>
            <button
              onClick={handleTrain}
              disabled={training}
              className={`btn-primary ${training ? 'pulsing' : ''}`}
            >
              {training ? 'Training...' : 'Start Fine-Tuning'}
            </button>
          </div>

          <div className="card">
            <h3>Model Parameters</h3>
            <div className="slider-group">
              <label>Temperature: {temperature}</label>
              <input
                type="range" min="0.1" max="1.0" step="0.1"
                value={temperature} onChange={(e) => setTemperature(e.target.value)}
              />
            </div>
            <div className="slider-group">
              <label>Max Tokens: {maxTokens}</label>
              <input
                type="range" min="50" max="1024" step="10"
                value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)}
              />
            </div>
            <div className="slider-group">
              <label>Top P: {topP}</label>
              <input
                type="range" min="0.1" max="1.0" step="0.05"
                value={topP} onChange={(e) => setTopP(e.target.value)}
              />
            </div>
            <div className="slider-group">
              <label>Rep. Penalty: {repPenalty}</label>
              <input
                type="range" min="1.0" max="2.0" step="0.1"
                value={repPenalty} onChange={(e) => setRepPenalty(e.target.value)}
              />
            </div>
          </div>
        </div>

        <div className="chat-interface">
          <div className="chat-history">
            {messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role}`}>
                <div className="message-bubble">{msg.content}</div>
              </div>
            ))}
            {loading && <div className="message assistant"><div className="message-bubble typing">Thinking...</div></div>}
          </div>

          <div className="input-area">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Ask a legal question about maritime jurisdiction..."
            />
            <button onClick={handleSend} className="btn-send">➤</button>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
