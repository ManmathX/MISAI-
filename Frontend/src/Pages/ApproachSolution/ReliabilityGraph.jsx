import React, { useEffect, useState } from 'react';
import './ReliabilityGraph.css';

const ReliabilityGraph = () => {
    const [isVisible, setIsVisible] = useState(false);

    useEffect(() => {
        // Trigger animation on mount
        const timer = setTimeout(() => setIsVisible(true), 100);
        return () => clearTimeout(timer);
    }, []);

    const metrics = [
        { name: 'Accuracy', standard: 65, misai: 98, color: '#ff0080' },
        { name: 'Reasoning', standard: 55, misai: 95, color: '#ff8c00' },
        { name: 'Consistency', standard: 45, misai: 92, color: '#40e0d0' },
        { name: 'Verification', standard: 30, misai: 99, color: '#9370db' }
    ];

    return (
        <div className="reliability-graph-container">
            <h3 className="graph-title rgb-text">Performance Comparison</h3>
            <p className="graph-subtitle">Standard AI vs. MISAI Multi-Agent System</p>

            <div className="graph-wrapper">
                <div className="y-axis">
                    <span>100%</span>
                    <span>75%</span>
                    <span>50%</span>
                    <span>25%</span>
                    <span>0%</span>
                </div>

                <div className="bars-container">
                    {metrics.map((metric, index) => (
                        <div key={index} className="metric-group">
                            <div className="bars-wrapper">
                                {/* Standard AI Bar */}
                                <div className="bar-column">
                                    <div
                                        className="bar standard-bar"
                                        style={{
                                            height: isVisible ? `${metric.standard}%` : '0%',
                                            transitionDelay: `${index * 0.1}s`
                                        }}
                                    >
                                        <span className="bar-value">{metric.standard}%</span>
                                    </div>
                                </div>

                                {/* MISAI Bar */}
                                <div className="bar-column">
                                    <div
                                        className="bar misai-bar rgb-border"
                                        style={{
                                            height: isVisible ? `${metric.misai}%` : '0%',
                                            backgroundColor: metric.color,
                                            boxShadow: `0 0 20px ${metric.color}66`,
                                            transitionDelay: `${index * 0.1 + 0.2}s`
                                        }}
                                    >
                                        <span className="bar-value">{metric.misai}%</span>
                                    </div>
                                </div>
                            </div>
                            <div className="x-label">{metric.name}</div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="graph-legend">
                <div className="legend-item">
                    <span className="legend-color standard-color"></span>
                    <span>Standard Single Model</span>
                </div>
                <div className="legend-item">
                    <span className="legend-color misai-color rgb-border"></span>
                    <span>MISAI Architecture</span>
                </div>
            </div>
        </div>
    );
};

export default ReliabilityGraph;
