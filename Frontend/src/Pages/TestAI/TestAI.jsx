import React, { useState } from 'react'
import './TestAI.css'
import Navbar from '../Navbar/Navbar'

const TestAI = () => {
  const [promptId, setPromptId] = useState('');
  const [userPrompt, setUserPrompt] = useState('');
  const [targetOutput, setTargetOutput] = useState('');
  const [groundTruth, setGroundTruth] = useState('');
  const [taskType, setTaskType] = useState('qa');
  const [language, setLanguage] = useState('en');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const API_BASE_URL = import.meta.env.VITE_HOST_URL || 'http://localhost:8000';

  const generateId = () => {
    return `eval_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };

  const getSeverityClass = (percentage) => {
    if (percentage < 20) return 'low';
    if (percentage < 50) return 'moderate';
    return 'high';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    setLoading(true);

    try {
      const payload = {
        prompt_id: promptId || generateId(),
        user_prompt: userPrompt,
        target_output: targetOutput,
        metadata: {
          task_type: taskType,
          language: language,
          eval_purpose: 'safety_and_quality'
        }
      };

      if (groundTruth.trim()) {
        payload.ground_truth = {
          type: 'text',
          content: groundTruth,
          sources: []
        };
      }

      const response = await fetch(`${API_BASE_URL}/evaluate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert(`Evaluation failed: ${error.message}`);
      console.error('Evaluation error:', error);
    } finally {
      setLoading(false);
    }
  };

  const renderSegment = (segment) => {
    let className = 'correct';
    if (segment.is_safety_violation) className = 'safety-violation';
    else if (segment.is_hallucination) className = 'hallucination';
    else if (segment.is_potential_fake_news) className = 'fake-news';

    return (
      <span
        key={segment.text + Math.random()}
        className={`segment ${className}`}
        title={segment.label || ''}
      >
        {segment.text || ''}
      </span>
    );
  };

  return (
    <>
      <Navbar />

      <div className="testai-container">
        {/* Header */}
        <header className="header">
          <div className="header-content">
            <h1 className="header-title">mr mama</h1>
            <p className="header-subtitle">its check other llm response</p>
          </div>
        </header>

        {/* Main Content */}
        <main className="main-content">
          {/* Tabs */}
          <div className="tabs">
            <button className="tab-btn active" data-tab="evaluate">Evaluate</button>
          </div>

          {/* Evaluate Tab */}
          <div className="tab-content active" id="evaluate-tab">
            <div className="card">
              <h2 className="card-title">Evaluate LLM Output</h2>

              <form onSubmit={handleSubmit}>
                <div className="form-group">
                  <label htmlFor="promptId">Prompt ID</label>
                  <input
                    type="text"
                    id="promptId"
                    value={promptId}
                    onChange={(e) => setPromptId(e.target.value)}
                    placeholder="Auto-generated if empty"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="userPrompt">User Prompt *</label>
                  <textarea
                    id="userPrompt"
                    rows={3}
                    required
                    value={userPrompt}
                    onChange={(e) => setUserPrompt(e.target.value)}
                    placeholder="Enter the original prompt..."
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="targetOutput">Target LLM Output *</label>
                  <textarea
                    id="targetOutput"
                    rows={6}
                    required
                    value={targetOutput}
                    onChange={(e) => setTargetOutput(e.target.value)}
                    placeholder="Enter the LLM output to evaluate..."
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="groundTruth">Ground Truth (Optional)</label>
                  <textarea
                    id="groundTruth"
                    rows={3}
                    value={groundTruth}
                    onChange={(e) => setGroundTruth(e.target.value)}
                    placeholder="Enter verified facts or sources..."
                  />
                </div>

                <div className="form-row">
                  <div className="form-group">
                    <label htmlFor="taskType">Task Type</label>
                    <select
                      id="taskType"
                      value={taskType}
                      onChange={(e) => setTaskType(e.target.value)}
                    >
                      <option value="qa">Q&A</option>
                      <option value="summarization">Summarization</option>
                      <option value="coding">Coding</option>
                      <option value="reasoning">Reasoning</option>
                      <option value="creative">Creative</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label htmlFor="language">Language</label>
                    <input
                      type="text"
                      id="language"
                      value={language}
                      onChange={(e) => setLanguage(e.target.value)}
                    />
                  </div>
                </div>

                <button type="submit" className="btn btn-primary" disabled={loading}>
                  {loading ? (
                    <>
                      <span className="loader"></span>
                      Evaluating...
                    </>
                  ) : (
                    'Evaluate Output'
                  )}
                </button>
              </form>

              {/* Results Display */}
              {result && (
                <div className="result-container">
                  <h3>Evaluation Results</h3>
                  <div className="result-content">
                    <div className="score-grid">
                      <div className="score-card">
                        <h4>Hallucination</h4>
                        <div className={`score-value ${getSeverityClass(result.judge_output.hallucination_probability_pct)}`}>
                          {result.judge_output.hallucination_probability_pct.toFixed(1)}%
                        </div>
                        <div className="score-bar">
                          <div
                            className={`score-bar-fill ${getSeverityClass(result.judge_output.hallucination_probability_pct)}`}
                            style={{ width: `${result.judge_output.hallucination_probability_pct}%` }}
                          ></div>
                        </div>
                      </div>

                      <div className="score-card">
                        <h4>Jailbreak</h4>
                        <div className={`score-value ${getSeverityClass(result.judge_output.jailbreak_probability_pct)}`}>
                          {result.judge_output.jailbreak_probability_pct.toFixed(1)}%
                        </div>
                        <div className="score-bar">
                          <div
                            className={`score-bar-fill ${getSeverityClass(result.judge_output.jailbreak_probability_pct)}`}
                            style={{ width: `${result.judge_output.jailbreak_probability_pct}%` }}
                          ></div>
                        </div>
                      </div>

                      <div className="score-card">
                        <h4>Fake News</h4>
                        <div className={`score-value ${getSeverityClass(result.judge_output.fake_news_probability_pct)}`}>
                          {result.judge_output.fake_news_probability_pct.toFixed(1)}%
                        </div>
                        <div className="score-bar">
                          <div
                            className={`score-bar-fill ${getSeverityClass(result.judge_output.fake_news_probability_pct)}`}
                            style={{ width: `${result.judge_output.fake_news_probability_pct}%` }}
                          ></div>
                        </div>
                      </div>

                      <div className="score-card">
                        <h4>Wrong Output</h4>
                        <div className={`score-value ${getSeverityClass(result.judge_output.wrong_output_probability_pct)}`}>
                          {result.judge_output.wrong_output_probability_pct.toFixed(1)}%
                        </div>
                        <div className="score-bar">
                          <div
                            className={`score-bar-fill ${getSeverityClass(result.judge_output.wrong_output_probability_pct)}`}
                            style={{ width: `${result.judge_output.wrong_output_probability_pct}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>

                    <div style={{ marginTop: '1.5rem' }}>
                      <h4 style={{ color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>Analysis</h4>
                      <p style={{ color: 'var(--text-primary)', lineHeight: '1.8' }}>
                        {result.judge_output.analysis_reasoning}
                      </p>
                    </div>

                    <div style={{ marginTop: '1.5rem' }}>
                      <h4 style={{ color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>Token Analysis</h4>
                      <p style={{ color: 'var(--text-primary)' }}>
                        Total Tokens: <strong>{result.total_output_tokens}</strong> |
                        Estimated Hallucinated: <strong>{result.estimated_hallucinated_tokens}</strong>
                        (<strong>{(result.judge_output.hallucination_token_fraction_estimate * 100).toFixed(1)}%</strong>)
                      </p>
                    </div>

                    <div className="segments-container">
                      <h4 style={{ color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>Segment Analysis</h4>
                      <div style={{ lineHeight: '2' }}>
                        {result.judge_output.segment_labels.map(seg => renderSegment(seg))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    </>
  );
};

export default TestAI;
