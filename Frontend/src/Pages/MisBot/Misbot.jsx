import React, { useState, useRef, useEffect } from 'react';
import './MisBot.css';
import Navbar from '../Navbar/Navbar';

const MisBot = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm MisBot. Ask me anything, and I'll synthesize the best answer from multiple AI models.",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSessionActive, setIsSessionActive] = useState(true);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleInputChange = (e) => {
    setInputMessage(e.target.value);
  };

  const startNewChat = () => {
    setMessages([
      {
        id: 1,
        text: "Hello! I'm MisBot. Ask me anything, and I'll synthesize the best answer from multiple AI models.",
        sender: 'bot',
        timestamp: new Date()
      }
    ]);
    setIsSessionActive(true);
    setInputMessage('');
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const userMessage = {
      id: messages.length + 1,
      text: inputMessage,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    setIsSessionActive(false);

    try {
      const response = await fetch(`${import.meta.env.VITE_HOST_URL}/chatapi`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage.text }),
      });

      if (!response.ok) throw new Error(`Error: ${response.status}`);

      const data = await response.json();

      const botResponse = {
        id: messages.length + 2,
        text: data.best_answer || "I'm having trouble processing your request.",
        sender: 'bot',
        timestamp: new Date(),
        rankedModels: data.ranked_models || [],
        summary: data.summary
      };

      setMessages(prevMessages => [...prevMessages, botResponse]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: messages.length + 2,
        text: "I'm sorry, I couldn't connect to the server. Please try again later.",
        sender: 'bot',
        timestamp: new Date(),
        isError: true
      };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
      setIsSessionActive(true);
    } finally {
      setIsLoading(false);
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getScoreColor = (score) => {
    if (score >= 0.8) return '#00e676';
    if (score >= 0.6) return '#ffd700';
    if (score >= 0.4) return '#ff9800';
    return '#ff4d4d';
  };

  const getScoreLabel = (score) => {
    if (score >= 0.8) return 'Excellent';
    if (score >= 0.6) return 'Good';
    if (score >= 0.4) return 'Fair';
    return 'Poor';
  };

  const MetricCard = ({ title, value, max = 1, icon }) => (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>
      <div className="metric-info">
        <div className="metric-title">{title}</div>
        <div className="metric-value-container">
          <div className="metric-value">{(value * 100).toFixed(0)}%</div>
          <div className="metric-label" style={{ color: getScoreColor(value) }}>
            {getScoreLabel(value)}
          </div>
        </div>
      </div>
      <div className="metric-bar">
        <div
          className="metric-bar-fill"
          style={{
            width: `${(value / max) * 100}%`,
            background: `linear-gradient(90deg, ${getScoreColor(value)}44, ${getScoreColor(value)})`
          }}
        />
      </div>
    </div>
  );

  return (
    <>
      <Navbar />
      <div className="misbot-container">
        <div className="chat-header">
          <h1>MisBot</h1>
          <p className="description">Multi-Model AI Synthesis & Verification</p>
        </div>

        <div className="chat-interface">
          <div className="messages-container">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'} ${message.isError ? 'error-message' : ''}`}
              >
                <div className="message-content">
                  <div className="text-content" style={{ whiteSpace: 'pre-wrap' }}>{message.text}</div>

                  {message.rankedModels && message.rankedModels.length > 0 && (
                    <div className="metrics-dashboard-new">
                      <div className="dashboard-header">
                        <h3>🎯 AI Reliability Analysis</h3>
                        <div className="consensus-badge" style={{
                          background: message.rankedModels[0].cacePerQuery > 1.5 ? 'linear-gradient(135deg, #ff4d4d22, #ff4d4d44)' : 'linear-gradient(135deg, #00e67622, #00e67644)',
                          border: `1px solid ${message.rankedModels[0].cacePerQuery > 1.5 ? '#ff4d4d' : '#00e676'}`
                        }}>
                          {message.rankedModels[0].cacePerQuery > 1.5 ? '⚠️ High Confusion' : '✓ High Consensus'}
                        </div>
                      </div>

                      {/* Key Metrics Grid */}
                      <div className="metrics-grid">
                        <MetricCard
                          title="Factual Accuracy"
                          value={message.rankedModels[0].factualAccuracy_grounded}
                          icon=""
                        />
                        <MetricCard
                          title="Reasoning Depth"
                          value={message.rankedModels[0].reasoningDepth}
                          icon=""
                        />
                        <MetricCard
                          title="Consistency"
                          value={message.rankedModels[0].consistency}
                          icon=""
                        />
                        <MetricCard
                          title="External Verification"
                          value={message.rankedModels[0].external_confidence}
                          icon=""
                        />
                      </div>

                      {/* Top Model Highlight */}
                      <div className="top-model-card">
                        <div className="top-model-badge">Best Model</div>
                        <div className="top-model-name">{message.rankedModels[0].modelName}</div>
                        <div className="top-model-score">
                          <span className="score-label">CARS Score:</span>
                          <span className="score-value" style={{ color: getScoreColor(message.rankedModels[0].CARS) }}>
                            {(message.rankedModels[0].CARS * 100).toFixed(1)}
                          </span>
                        </div>
                      </div>

                      {/* Model Rankings Table */}
                      <div className="rankings-section">
                        <h4>Model Rankings</h4>
                        <div className="rankings-table">
                          <div className="rankings-header">
                            <span className="rank-col">Rank</span>
                            <span className="model-col">Model</span>
                            <span className="score-col">CARS Score</span>
                            <span className="status-col">Status</span>
                          </div>
                          {message.rankedModels.map((model, index) => (
                            <div key={index} className={`ranking-row ${index === 0 ? 'top-rank' : ''}`}>
                              <span className="rank-col">
                                {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : `#${index + 1}`}
                              </span>
                              <span className="model-col">{model.modelName}</span>
                              <span className="score-col">
                                <div className="score-bar-mini">
                                  <div
                                    className="score-bar-fill-mini"
                                    style={{
                                      width: `${model.CARS * 100}%`,
                                      background: getScoreColor(model.CARS)
                                    }}
                                  />
                                </div>
                                <span style={{ color: getScoreColor(model.CARS) }}>
                                  {(model.CARS * 100).toFixed(1)}
                                </span>
                              </span>
                              <span className="status-col">
                                <span className="status-badge" style={{
                                  background: `${getScoreColor(model.CARS)}22`,
                                  color: getScoreColor(model.CARS),
                                  border: `1px solid ${getScoreColor(model.CARS)}`
                                }}>
                                  {getScoreLabel(model.CARS)}
                                </span>
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Detailed Metrics Comparison */}
                      <div className="detailed-metrics">
                        <h4>Detailed Comparison</h4>
                        <div className="comparison-grid">
                          {['factualAccuracy_grounded', 'reasoningDepth', 'consistency', 'sourceVerification'].map((metric) => {
                            const metricNames = {
                              factualAccuracy_grounded: 'Factual Accuracy',
                              reasoningDepth: 'Reasoning Depth',
                              consistency: 'Consistency',
                              sourceVerification: 'Source Verification'
                            };
                            const topValue = message.rankedModels[0][metric];
                            const avgValue = message.rankedModels.reduce((acc, m) => acc + m[metric], 0) / message.rankedModels.length;

                            return (
                              <div key={metric} className="comparison-item">
                                <div className="comparison-title">{metricNames[metric]}</div>
                                <div className="comparison-bars">
                                  <div className="comparison-bar-row">
                                    <span className="bar-label">Best</span>
                                    <div className="comparison-bar">
                                      <div
                                        className="comparison-bar-fill"
                                        style={{
                                          width: `${topValue * 100}%`,
                                          background: '#00E5FF'
                                        }}
                                      />
                                    </div>
                                    <span className="bar-value">{(topValue * 100).toFixed(0)}%</span>
                                  </div>
                                  <div className="comparison-bar-row">
                                    <span className="bar-label">Avg</span>
                                    <div className="comparison-bar">
                                      <div
                                        className="comparison-bar-fill"
                                        style={{
                                          width: `${avgValue * 100}%`,
                                          background: '#FFD700'
                                        }}
                                      />
                                    </div>
                                    <span className="bar-value">{(avgValue * 100).toFixed(0)}%</span>
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                <span className="message-time">{formatTimestamp(message.timestamp)}</span>
              </div>
            ))}
            {isLoading && (
              <div className="message bot-message loading-message">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-area">
            {!isSessionActive && !isLoading ? (
              <button onClick={startNewChat} className="restart-button">
                Start New Chat
              </button>
            ) : (
              <form onSubmit={sendMessage} className="message-input-form">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={handleInputChange}
                  placeholder="Type your query here..."
                  disabled={isLoading || !isSessionActive}
                  className="message-input"
                />
                <button
                  type="submit"
                  disabled={isLoading || !inputMessage.trim()}
                  className="send-button"
                >
                  Send
                </button>
              </form>
            )}
          </div>
        </div>
      </div>
    </>
  );
};

export default MisBot;