import React, { useEffect, useState } from 'react';
import './CARSGraph.css';

const CARSGraph = () => {
    const [isVisible, setIsVisible] = useState(false);

    useEffect(() => {
        // Trigger animation on mount
        const timer = setTimeout(() => setIsVisible(true), 100);
        return () => clearTimeout(timer);
    }, []);

    // Define the CARS components with weights and colors
    const components = [
        { name: 'Accuracy (α)', symbol: 'A', weight: 0.3, value: 85, color: '#ff0080' },
        { name: 'Reasoning (β)', symbol: 'R', weight: 0.25, value: 90, color: '#ff8c00' },
        { name: 'Consistency (γ)', symbol: 'C', weight: 0.25, value: 88, color: '#40e0d0' },
        { name: 'Verification (δ)', symbol: 'V', weight: 0.2, value: 95, color: '#9370db' }
    ];

    // Calculate weighted scores
    const weightedScores = components.map(comp => ({
        ...comp,
        weighted: comp.weight * comp.value
    }));

    const totalScore = weightedScores.reduce((sum, comp) => sum + comp.weighted, 0).toFixed(1);

    return (
        <div className="cars-graph-container">
            <h4 className="cars-graph-title">CARS Component Breakdown</h4>
            <p className="cars-graph-subtitle">Weighted contributions to overall reliability score</p>

            {/* Stacked horizontal bar */}
            <div className="cars-stacked-bar-wrapper">
                <div className="cars-stacked-bar">
                    {weightedScores.map((comp, index) => (
                        <div
                            key={index}
                            className="cars-bar-segment"
                            style={{
                                width: isVisible ? `${(comp.weighted / totalScore) * 100}%` : '0%',
                                backgroundColor: comp.color,
                                boxShadow: `0 0 15px ${comp.color}66`,
                                transitionDelay: `${index * 0.15}s`
                            }}
                            title={`${comp.name}: ${comp.weighted.toFixed(1)}`}
                        >
                            <span className="segment-label">{comp.symbol}</span>
                        </div>
                    ))}
                </div>
                <div className="cars-total-score">
                    <span className="score-label">CARS Score:</span>
                    <span className="score-value rgb-text">{totalScore}</span>
                </div>
            </div>

            {/* Component details */}
            <div className="cars-components-grid">
                {weightedScores.map((comp, index) => (
                    <div
                        key={index}
                        className="cars-component-card"
                        style={{
                            opacity: isVisible ? 1 : 0,
                            transform: isVisible ? 'translateY(0)' : 'translateY(20px)',
                            transitionDelay: `${index * 0.1 + 0.3}s`
                        }}
                    >
                        <div className="component-header">
                            <div
                                className="component-color-indicator"
                                style={{ backgroundColor: comp.color, boxShadow: `0 0 10px ${comp.color}` }}
                            ></div>
                            <span className="component-name">{comp.name}</span>
                        </div>
                        <div className="component-stats">
                            <div className="stat-row">
                                <span className="stat-label">Weight:</span>
                                <span className="stat-value">{comp.weight}</span>
                            </div>
                            <div className="stat-row">
                                <span className="stat-label">Score:</span>
                                <span className="stat-value">{comp.value}</span>
                            </div>
                            <div className="stat-row highlight">
                                <span className="stat-label">Contribution:</span>
                                <span className="stat-value" style={{ color: comp.color }}>
                                    {comp.weighted.toFixed(1)}
                                </span>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Formula reminder */}
            <div className="cars-formula-reminder">
                <p>
                    The total CARS score combines all weighted components where weights sum to 1.0
                </p>
            </div>
        </div>
    );
};

export default CARSGraph;
