import React, { useEffect, useState } from 'react';
import './CACEGraph.css';

const CACEGraph = () => {
    const [isVisible, setIsVisible] = useState(false);
    const [activeScenario, setActiveScenario] = useState('low'); // 'low' or 'high' entropy

    useEffect(() => {
        // Trigger animation on mount
        const timer = setTimeout(() => setIsVisible(true), 100);
        return () => clearTimeout(timer);
    }, []);

    // Define scenarios
    const scenarios = {
        low: {
            title: 'Low Entropy (High Agreement)',
            description: 'AI models produce very similar responses',
            models: [
                { id: 1, name: 'GPT-4', distance: 0.05, x: 150, y: 100 },
                { id: 2, name: 'Claude', distance: 0.08, x: 250, y: 80 },
                { id: 3, name: 'Gemini', distance: 0.06, x: 200, y: 140 },
                { id: 4, name: 'Perplexity', distance: 0.07, x: 300, y: 120 }
            ],
            entropy: 0.42,
            color: '#40e0d0'
        },
        high: {
            title: 'High Entropy (Low Agreement)',
            description: 'AI models produce divergent responses',
            models: [
                { id: 1, name: 'GPT-4', distance: 0.15, x: 100, y: 50 },
                { id: 2, name: 'Claude', distance: 0.45, x: 350, y: 80 },
                { id: 3, name: 'Gemini', distance: 0.30, x: 150, y: 180 },
                { id: 4, name: 'Perplexity', distance: 0.50, x: 320, y: 160 }
            ],
            entropy: 1.38,
            color: '#ff0080'
        }
    };

    const currentScenario = scenarios[activeScenario];

    // Calculate similarity scores and probabilities
    const modelsWithScores = currentScenario.models.map(model => {
        const similarity = Math.exp(-model.distance);
        return { ...model, similarity };
    });

    const totalSimilarity = modelsWithScores.reduce((sum, m) => sum + m.similarity, 0);
    const modelsWithProbs = modelsWithScores.map(model => ({
        ...model,
        probability: model.similarity / totalSimilarity
    }));

    return (
        <div className="cace-graph-container">
            <h4 className="cace-graph-title">CACE Entropy Visualization</h4>
            <p className="cace-graph-subtitle">Interactive model agreement demonstration</p>

            {/* Scenario Toggle */}
            <div className="scenario-toggle">
                <button
                    className={`scenario-btn ${activeScenario === 'low' ? 'active' : ''}`}
                    onClick={() => setActiveScenario('low')}
                >
                    High Agreement
                </button>
                <button
                    className={`scenario-btn ${activeScenario === 'high' ? 'active' : ''}`}
                    onClick={() => setActiveScenario('high')}
                >
                    Low Agreement
                </button>
            </div>

            {/* Scenario Info */}
            <div
                className="scenario-info"
                style={{
                    borderColor: currentScenario.color,
                    opacity: isVisible ? 1 : 0
                }}
            >
                <h5 style={{ color: currentScenario.color }}>{currentScenario.title}</h5>
                <p>{currentScenario.description}</p>
                <div className="entropy-display">
                    <span className="entropy-label">CACE Score:</span>
                    <span className="entropy-value" style={{ color: currentScenario.color }}>
                        {currentScenario.entropy.toFixed(2)}
                    </span>
                </div>
            </div>

            {/* Visual Representation */}
            <div className="cace-visualization">
                <svg className="model-graph" viewBox="0 0 400 200">
                    {/* Central consolidated answer node */}
                    <circle
                        cx="200"
                        cy="100"
                        r="20"
                        fill="rgba(255, 255, 255, 0.2)"
                        stroke="#40e0d0"
                        strokeWidth="2"
                        className={isVisible ? 'fade-in' : ''}
                    />
                    <text
                        x="200"
                        y="105"
                        textAnchor="middle"
                        fill="#40e0d0"
                        fontSize="12"
                        fontWeight="700"
                    >
                        Consensus
                    </text>

                    {/* Model nodes and connections */}
                    {modelsWithProbs.map((model, idx) => (
                        <g key={model.id}>
                            {/* Connection line */}
                            <line
                                x1="200"
                                y1="100"
                                x2={model.x}
                                y2={model.y}
                                stroke={currentScenario.color}
                                strokeWidth={model.probability * 4}
                                opacity={isVisible ? 0.6 : 0}
                                className="connection-line"
                                style={{ transitionDelay: `${idx * 0.1}s` }}
                            />

                            {/* Model node */}
                            <circle
                                cx={model.x}
                                cy={model.y}
                                r={10 + model.probability * 15}
                                fill={currentScenario.color}
                                opacity={isVisible ? 0.8 : 0}
                                className="model-node"
                                style={{ transitionDelay: `${idx * 0.15}s` }}
                            />

                            {/* Model label */}
                            <text
                                x={model.x}
                                y={model.y - 20}
                                textAnchor="middle"
                                fill="#e5e7eb"
                                fontSize="10"
                                fontWeight="600"
                                opacity={isVisible ? 1 : 0}
                                style={{ transitionDelay: `${idx * 0.15}s` }}
                            >
                                {model.name}
                            </text>
                        </g>
                    ))}
                </svg>
            </div>

            {/* Model Details Table */}
            <div className="models-table">
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Distance (d<sub>i</sub>)</th>
                            <th>Similarity (s<sub>i</sub>)</th>
                            <th>Probability (p<sub>i</sub>)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {modelsWithProbs.map((model, idx) => (
                            <tr
                                key={model.id}
                                style={{
                                    opacity: isVisible ? 1 : 0,
                                    transform: isVisible ? 'translateX(0)' : 'translateX(-20px)',
                                    transitionDelay: `${idx * 0.1 + 0.3}s`
                                }}
                            >
                                <td className="model-name">{model.name}</td>
                                <td>{model.distance.toFixed(2)}</td>
                                <td>{model.similarity.toFixed(3)}</td>
                                <td>
                                    <div className="probability-bar-container">
                                        <div
                                            className="probability-bar"
                                            style={{
                                                width: `${model.probability * 100}%`,
                                                backgroundColor: currentScenario.color
                                            }}
                                        ></div>
                                        <span className="probability-text">
                                            {(model.probability * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Interpretation */}
            <div className="cace-interpretation">
                <p>
                    {activeScenario === 'low' ? (
                        <>
                            <strong style={{ color: '#40e0d0' }}>Low CACE score</strong> indicates high agreement among models,
                            suggesting a reliable and consistent response.
                        </>
                    ) : (
                        <>
                            <strong style={{ color: '#ff0080' }}>High CACE score</strong> indicates disagreement among models,
                            signaling potential uncertainty or complexity in the query.
                        </>
                    )}
                </p>
            </div>
        </div>
    );
};

export default CACEGraph;
