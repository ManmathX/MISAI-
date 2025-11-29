import React, { useState } from 'react'
import './TestImage.css'
import Navbar from '../Navbar/Navbar'

const TestImage = () => {
  const [selectedImage, setSelectedImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [statusMessage, setStatusMessage] = useState('')
  const [error, setError] = useState(null)

  const wait = (ms) => new Promise(res => setTimeout(res, ms))

  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (!file) return

    setSelectedImage(file)
    setImagePreview(URL.createObjectURL(file))
    setResults(null)
    setError(null)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!selectedImage) {
      setError('Please select an image to verify')
      return
    }

    setLoading(true)
    setStatusMessage('Uploading image...')
    setResults(null)
    setError(null)

    try {
      // STEP 1: Upload
      const formData = new FormData()
      formData.append('file', selectedImage)

      const uploadResponse = await fetch(`${import.meta.env.VITE_BACKEND_HOST_URL}/upload`, {
        method: 'POST',
        body: formData
      })

      if (!uploadResponse.ok) {
        const err = await uploadResponse.text()
        throw new Error(`Upload failed: ${err}`)
      }

      const uploadData = await uploadResponse.json()
      const requestId = uploadData.requestId
      if (!requestId) throw new Error("Request ID missing")

      // STEP 2: Polling
      setStatusMessage('Analyzing deepfake models...')

      const COMPLETE_STATES = ['COMPLETED', 'DONE', 'AUTHENTIC', 'MANIPULATED', 'FAKE']
      let analysisData = null

      for (let i = 0; i < 30; i++) {
        await wait(2000)

        const resultResponse = await fetch(`${import.meta.env.VITE_BACKEND_HOST_URL}/result/${requestId}`)
        if (!resultResponse.ok) continue

        const data = await resultResponse.json()

        if (COMPLETE_STATES.includes(data.overallStatus)) {
          analysisData = data
          break
        }

        if (data.overallStatus === 'FAILED') {
          throw new Error("Analysis failed on server")
        }
      }

      if (!analysisData) throw new Error("Analysis timed out")

      setResults(analysisData)

    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
      setStatusMessage('')
    }
  }

  const renderVerificationResult = () => {
    if (!results) return null

    const models = results.deepfakeModels || []

    // Calculate Average Score (only for models with numeric scores)
    const validModels = models.filter(m => typeof m.score === 'number')
    const totalScore = validModels.reduce((sum, m) => sum + m.score, 0)
    const averageScore = validModels.length > 0 ? totalScore / validModels.length : 0
    const averagePercentage = (averageScore * 100).toFixed(1)

    // Determine status based on average or overallStatus
    // If overallStatus is explicitly FAKE/MANIPULATED, trust it.
    const isFake = averageScore > 0.5 || results.overallStatus === 'MANIPULATED' || results.overallStatus === 'FAKE'
    const statusColor = isFake ? '#ff4444' : '#00c851'
    const statusText = isFake ? 'Potential Deepfake Detected' : 'Likely Authentic Image'

    return (
      <div className="verification-result">

        {/* HEADER & SUMMARY */}
        <div className="result-header" style={{ borderColor: statusColor }}>
          <h2 style={{ color: statusColor }}>{statusText}</h2>
          <div className="score-overview">
            <div className="score-circle" style={{ borderColor: statusColor }}>
              <span className="score-value">{averagePercentage}%</span>
              <span className="score-label">Avg. Fake Probability</span>
            </div>
            <div className="meta-score">
              <span className="label">Final Score</span>
              <span className="value">{results.resultsSummary?.metadata?.finalScore || 'N/A'}</span>
            </div>
          </div>
        </div>

        {/* MODEL BREAKDOWN GRID */}
        {models.length > 0 && (
          <div className="models-section">
            <h3>Model Analysis</h3>
            <div className="models-grid">
              {models.map((model, idx) => {
                const isNumericScore = typeof model.score === 'number'
                const modelScorePct = isNumericScore ? (model.score * 100).toFixed(1) : 'N/A'
                const modelIsFake = isNumericScore && model.score > 0.5

                return (
                  <div key={idx} className="model-card">
                    <div className="model-name">{model.name}</div>

                    {isNumericScore ? (
                      <>
                        <div className="model-score-bar">
                          <div
                            className="bar-fill"
                            style={{
                              width: `${modelScorePct}%`,
                              backgroundColor: modelIsFake ? '#ff4444' : '#00c851'
                            }}
                          ></div>
                        </div>
                        <div className="model-score-text">
                          Score: {model.score.toFixed(4)} ({modelScorePct}%)
                        </div>
                      </>
                    ) : (
                      <div className="model-score-text" style={{ marginTop: '1rem', fontStyle: 'italic' }}>
                        Not Evaluated / Not Applicable
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        )}

      </div>
    )
  }

  return (
    <>
      <Navbar />

      <div className="testimage-container">
        <h1>Image Verification Tool</h1>

        <div className={`image-verification-area ${results ? "with-results" : "no-results"}`}>
          <div className="upload-section">

            <form onSubmit={handleSubmit} className="upload-form">

              {/* IMAGE UPLOAD */}
              <div className="image-upload-container">
                {imagePreview ? (
                  <div className="image-preview-container">
                    <img src={imagePreview} alt="Preview" className="image-preview" />

                    <button
                      type="button"
                      className="change-image-btn"
                      onClick={() => {
                        setSelectedImage(null)
                        setImagePreview(null)
                        setResults(null)
                        setError(null)
                      }}
                    >
                      Change Image
                    </button>
                  </div>
                ) : (
                  <div className="upload-placeholder">
                    <label htmlFor="image-upload" className="upload-label">
                      <div className="upload-icon">
                        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48"
                          viewBox="0 0 24 24" fill="none" stroke="currentColor"
                          strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                          <circle cx="8.5" cy="8.5" r="1.5"></circle>
                          <polyline points="21 15 16 10 5 21"></polyline>
                        </svg>
                      </div>
                      <span>Click to upload an image</span>
                      <span className="upload-hint">JPG, PNG or GIF (max 10MB)</span>
                    </label>
                    <input
                      type="file"
                      id="image-upload"
                      accept="image/*"
                      onChange={handleImageChange}
                      className="file-input"
                    />
                  </div>
                )}
              </div>

              {/* VERIFY BTN */}
              <button
                type="submit"
                className="verify-button"
                disabled={loading || !selectedImage}
              >
                {loading ? "Processing..." : "Verify Image"}
              </button>
            </form>

            {/* RESULTS */}
            {(results || loading || error) && (
              <div className="results-section">

                {error && <div className="error-message">{error}</div>}

                {loading && (
                  <div className="loading-indicator">
                    <div className="spinner"></div>
                    <p>{statusMessage}</p>
                  </div>
                )}

                {!loading && renderVerificationResult()}
              </div>
            )}

          </div>
        </div>
      </div>
    </>
  )
}

export default TestImage
