import React, { useState, useEffect } from 'react';
import './CustomLossGraph.css';

const CustomLossGraph = () => {
    // Parameters
    const [lambda, setLambda] = useState(0.1);
    const [epsilon, setEpsilon] = useState(0.001);
    const [weights, setWeights] = useState([0.5, -0.3, 0.8, -0.2, 0.6]);

    // Calculated values
    const [Sj, setSj] = useState(0);
    const [Qj, setQj] = useState(0);
    const [Dj, setDj] = useState(0);
    const [Rj, setRj] = useState(0);
    const [penalty, setPenalty] = useState(0);

    // Calculate helper quantities whenever weights or epsilon change
    useEffect(() => {
        // S_j = sum of absolute values
        const s = weights.reduce((sum, w) => sum + Math.abs(w), 0);
        setSj(s);

        // Q_j = sum of squares
        const q = weights.reduce((sum, w) => sum + w * w, 0);
        setQj(q);

        // D_j = sqrt(Q_j + epsilon)
        const d = Math.sqrt(q + epsilon);
        setDj(d);

        // R_j = S_j / D_j
        const r = s / d;
        setRj(r);

        // P(W) = (R_j - 1)^2
        const p = Math.pow(r - 1, 2);
        setPenalty(p);
    }, [weights, epsilon]);

    // Update a specific weight
    const updateWeight = (index, value) => {
        const newWeights = [...weights];
        newWeights[index] = parseFloat(value) || 0;
        setWeights(newWeights);
    };

    // Generate penalty curve data for P(W) vs R_j plot
    const generatePenaltyCurve = () => {
        const points = [];
        for (let r = 0; r <= 3; r += 0.05) {
            const p = Math.pow(r - 1, 2);
            points.push({ r, p });
        }
        return points;
    };

    const penaltyCurve = generatePenaltyCurve();

    // SVG dimensions
    const width = 500;
    const height = 300;
    const padding = 50;
    const plotWidth = width - 2 * padding;
    const plotHeight = height - 2 * padding;

    // Scale functions
    const xScale = (r) => padding + (r / 3) * plotWidth;
    const yScale = (p) => height - padding - (p / 4) * plotHeight;

    // Generate path for curve
    const pathData = penaltyCurve
        .map((point, i) => {
            const x = xScale(point.r);
            const y = yScale(point.p);
            return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
        })
        .join(' ');

    return (
        <div className="custom-loss-graph">
            <div className="loss-content">

                {/* Formula Section */}
                <div className="formula-section">
                    <h3>Custom Balanced-Usage Loss</h3>
                    <div className="formula-box">
                        <p className="formula-text">
                            {"\\[ L(\\mathbf{W}; \\mathbf{x}, y) = (\\hat{y} - y)^2 + \\lambda \\sum_{j=1}^{m} (R_j - 1)^2 \\]"}
                        </p>
                        <p className="formula-text">
                            {"\\[ R_j = \\frac{S_j}{D_j} \\quad \\text{where} \\quad S_j = \\sum_{i=1}^{d} |W_{ij}|, \\quad D_j = \\sqrt{Q_j + \\varepsilon} \\]"}
                        </p>
                    </div>
                </div>

                {/* Two-column layout for graphs */}
                <div className="graphs-container">

                    {/* Left: Penalty Plot */}
                    <div className="graph-card">
                        <h4>Penalty Term: P(W) vs R<sub>j</sub></h4>
                        <svg width={width} height={height} className="penalty-plot">
                            {/* Grid lines */}
                            <g className="grid">
                                {[0, 1, 2, 3, 4].map(p => (
                                    <line
                                        key={`h-${p}`}
                                        x1={padding}
                                        y1={yScale(p)}
                                        x2={width - padding}
                                        y2={yScale(p)}
                                        stroke="rgba(255, 255, 255, 0.1)"
                                        strokeWidth="1"
                                    />
                                ))}
                                {[0, 0.5, 1, 1.5, 2, 2.5, 3].map(r => (
                                    <line
                                        key={`v-${r}`}
                                        x1={xScale(r)}
                                        y1={padding}
                                        x2={xScale(r)}
                                        y2={height - padding}
                                        stroke="rgba(255, 255, 255, 0.1)"
                                        strokeWidth="1"
                                    />
                                ))}
                            </g>

                            {/* Axes */}
                            <line
                                x1={padding}
                                y1={height - padding}
                                x2={width - padding}
                                y2={height - padding}
                                stroke="#fff"
                                strokeWidth="2"
                            />
                            <line
                                x1={padding}
                                y1={padding}
                                x2={padding}
                                y2={height - padding}
                                stroke="#fff"
                                strokeWidth="2"
                            />

                            {/* Penalty curve */}
                            <path
                                d={pathData}
                                fill="none"
                                stroke="url(#gradient-loss)"
                                strokeWidth="3"
                            />

                            {/* Current point */}
                            <circle
                                cx={xScale(Rj)}
                                cy={yScale(penalty)}
                                r="6"
                                fill="#ff0080"
                                className="current-point"
                            />

                            {/* Optimal point at R_j = 1 */}
                            <circle
                                cx={xScale(1)}
                                cy={yScale(0)}
                                r="5"
                                fill="#40e0d0"
                                className="optimal-point"
                            />
                            <text
                                x={xScale(1)}
                                y={yScale(0) - 15}
                                fill="#40e0d0"
                                fontSize="12"
                                textAnchor="middle"
                            >
                                Optimal (R<tspan baselineShift="sub" fontSize="10">j</tspan>=1)
                            </text>

                            {/* Axis labels */}
                            <text
                                x={width / 2}
                                y={height - 10}
                                fill="#fff"
                                fontSize="14"
                                textAnchor="middle"
                            >
                                R<tspan baselineShift="sub" fontSize="11">j</tspan> (Ratio)
                            </text>
                            <text
                                x={15}
                                y={height / 2}
                                fill="#fff"
                                fontSize="14"
                                textAnchor="middle"
                                transform={`rotate(-90, 15, ${height / 2})`}
                            >
                                P(W) = (R<tspan baselineShift="sub" fontSize="11">j</tspan> - 1)²
                            </text>

                            {/* Gradient definition */}
                            <defs>
                                <linearGradient id="gradient-loss" x1="0%" y1="0%" x2="100%" y2="0%">
                                    <stop offset="0%" stopColor="#ff0080" />
                                    <stop offset="33%" stopColor="#40e0d0" />
                                    <stop offset="66%" stopColor="#ff8c00" />
                                    <stop offset="100%" stopColor="#9370db" />
                                </linearGradient>
                            </defs>
                        </svg>
                    </div>

                    {/* Right: Interactive Controls */}
                    <div className="graph-card">
                        <h4>Interactive Loss Calculator</h4>

                        {/* Parameter Controls */}
                        <div className="controls-section">
                            <div className="control-group">
                                <label>
                                    λ (Lambda): <span className="param-value">{lambda.toFixed(3)}</span>
                                </label>
                                <input
                                    type="range"
                                    min="0"
                                    max="1"
                                    step="0.01"
                                    value={lambda}
                                    onChange={(e) => setLambda(parseFloat(e.target.value))}
                                    className="slider"
                                />
                            </div>

                            <div className="control-group">
                                <label>
                                    ε (Epsilon): <span className="param-value">{epsilon.toFixed(4)}</span>
                                </label>
                                <input
                                    type="range"
                                    min="0.0001"
                                    max="0.01"
                                    step="0.0001"
                                    value={epsilon}
                                    onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                                    className="slider"
                                />
                            </div>
                        </div>

                        {/* Weight Inputs */}
                        <div className="weights-section">
                            <h5>Neuron Weights (W<sub>ij</sub>)</h5>
                            <div className="weight-inputs">
                                {weights.map((weight, index) => (
                                    <div key={index} className="weight-input-group">
                                        <label>W<sub>{index + 1}j</sub></label>
                                        <input
                                            type="number"
                                            step="0.1"
                                            value={weight}
                                            onChange={(e) => updateWeight(index, e.target.value)}
                                            className="weight-input"
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Calculated Values Display */}
                        <div className="results-section">
                            <h5>Calculated Values</h5>
                            <div className="results-grid">
                                <div className="result-item">
                                    <span className="result-label">S<sub>j</sub></span>
                                    <span className="result-value">{Sj.toFixed(4)}</span>
                                </div>
                                <div className="result-item">
                                    <span className="result-label">Q<sub>j</sub></span>
                                    <span className="result-value">{Qj.toFixed(4)}</span>
                                </div>
                                <div className="result-item">
                                    <span className="result-label">D<sub>j</sub></span>
                                    <span className="result-value">{Dj.toFixed(4)}</span>
                                </div>
                                <div className="result-item highlight">
                                    <span className="result-label">R<sub>j</sub></span>
                                    <span className="result-value">{Rj.toFixed(4)}</span>
                                </div>
                                <div className="result-item highlight">
                                    <span className="result-label">P(W)</span>
                                    <span className="result-value">{penalty.toFixed(4)}</span>
                                </div>
                                <div className="result-item highlight">
                                    <span className="result-label">λ·P(W)</span>
                                    <span className="result-value">{(lambda * penalty).toFixed(4)}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Formula Explanation */}
                <div className="explanation-section">
                    <h4>Key Components</h4>
                    <div className="explanation-grid">
                        <div className="explanation-item">
                            <strong>S<sub>j</sub></strong>: L1-norm (sum of absolute weights)
                        </div>
                        <div className="explanation-item">
                            <strong>Q<sub>j</sub></strong>: L2-norm squared (sum of weight squares)
                        </div>
                        <div className="explanation-item">
                            <strong>D<sub>j</sub></strong>: Stabilized L2 norm
                        </div>
                        <div className="explanation-item">
                            <strong>R<sub>j</sub></strong>: Balance ratio (optimal at 1)
                        </div>
                        <div className="explanation-item">
                            <strong>P(W)</strong>: Penalty term encouraging balanced usage
                        </div>
                        <div className="explanation-item">
                            <strong>λ</strong>: Controls penalty strength in loss
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CustomLossGraph;
