import React, { useEffect } from 'react';
import ReliabilityGraph from './ReliabilityGraph';
import CARSGraph from './CARSGraph';
import CACEGraph from './CACEGraph';
import CustomLossGraph from './CustomLossGraph';
import './ApproachSolution.css';
import Navbar from '../Navbar/Navbar';
import Footer from '../Footer/Footer';

const ApproachSolution = () => {
    useEffect(() => {
        // Configure MathJax
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']]
            },
            svg: {
                fontCache: 'global'
            }
        };

        // Load MathJax script dynamically if not already loaded
        if (!window.MathJax.typeset) {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
            script.async = true;
            document.head.appendChild(script);
        } else {
            // If already loaded, re-render
            window.MathJax.typesetPromise && window.MathJax.typesetPromise();
        }
    }, []);

    return (
        <>
            <Navbar />
            <div className="approach-container">
                <div className="approach-header">
                    <h1 className="approach-title">Our Approach & Solution</h1>
                    <div className="title-underline"></div>
                </div>

                <div className="approach-section">
                    <h2>Platform Overview</h2>
                    <p>
                        MISAI is an advanced Multi-Agent System (MAS) designed to evaluate, verify, and consolidate responses from multiple Large Language Models (LLMs).
                        By leveraging a decentralized architecture of specialized AI agents, we provide a reliability layer that transcends the limitations of any single model.
                    </p>

                    <div className="features-grid">
                        <div className="feature-card">
                            <div className="feature-icon">⚡</div>
                            <h3>Parallel Model Querying</h3>
                            <p>Simultaneously queries top-tier models (GPT-4, Claude 3, Gemini 1.5) to gather diverse perspectives on complex queries.</p>
                        </div>

                        <div className="feature-card">
                            <div className="feature-icon">🔍</div>
                            <h3>Evidence Retrieval</h3>
                            <p>Autonomous agents cross-reference claims against trusted real-time data sources to verify factual accuracy.</p>
                        </div>

                        <div className="feature-card">
                            <div className="feature-icon">🔄</div>
                            <h3>Consolidated Response</h3>
                            <p>Synthesizes the best insights from all models into a single, high-confidence answer with cited sources.</p>
                        </div>

                        <div className="feature-card">
                            <div className="feature-icon">📊</div>
                            <h3>Interactive Dashboard</h3>
                            <p>Visualizes model agreement, confidence scores, and reasoning paths in real-time for full transparency.</p>
                        </div>

                        <div className="feature-card">
                            <div className="feature-icon">⚙️</div>
                            <h3>User Controls</h3>
                            <p>Customizable parameters for model selection, verification depth, and response format to suit your needs.</p>
                        </div>

                        <div className="feature-card">
                            <div className="feature-icon">📈</div>
                            <h3>Scalability</h3>
                            <p>Built on a modular architecture that easily scales to incorporate new models and verification tools.</p>
                        </div>
                    </div>
                </div>

                <div className="approach-section">
                    <ReliabilityGraph />
                </div>

                <div className="approach-section">
                    <h2>Mathematical Foundation</h2>
                    <p className="metrics-intro">
                        Two reliability metrics are proposed to provide a robust, measurable, and interpretable way to assess AI model performance and trustworthiness across diverse inputs:
                    </p>

                    <div className="metric-card">
                        <h3>Formula 1: Composite AI Reliability Score (CARS)</h3>
                        <p>
                            Define the Composite AI Reliability Score (CARS) for an AI model \(i\) across \(n\) queries as a weighted combination of four components: factual accuracy \(A_i\), reasoning depth \(R_i\), consistency \(C_i\), and source verification confidence \(V_i\).
                        </p>

                        <div className="formula-box" dangerouslySetInnerHTML={{ __html: '$$\\mathrm{CARS}(i) = \\alpha A_i + \\beta R_i + \\gamma C_i + \\delta V_i$$' }}></div>

                        <p>where \(\alpha, \beta, \gamma, \delta \in [0,1]\) are weights such that</p>

                        <div className="formula-box">
                            $$\alpha + \beta + \gamma + \delta = 1$$
                        </div>

                        <div className="metric-explanation">
                            <p>
                                This score combines multiple quality dimensions to provide a comprehensive assessment of AI reliability. Each component is weighted to reflect its importance in determining overall trustworthiness.
                            </p>
                        </div>

                        <CARSGraph />
                    </div>

                    <div className="metric-card">
                        <h3>Formula 2: Cross-AI Consistency Entropy (CACE)</h3>
                        <p>
                            The Cross-AI Consistency Entropy (CACE) calculates a score based on how similar each AI's response is to the overall best or combined answer. If the AI answers are very similar, the entropy (CACE) score is low, meaning they mostly agree. If their answers are very different, the score is high, meaning they disagree more. This helps in understanding the consistency or agreement among multiple AI responses for the same query.
                        </p>

                        <p>
                            To quantify the overall consistency between AI models' responses per query \(q\), define:
                        </p>

                        <div className="formula-box" dangerouslySetInnerHTML={{ __html: '$$\\mathrm{CACE}_q = - \\sum_{i=1}^{m} p_i \\log(p_i)$$' }}></div>

                        <p>where:</p>
                        <ul className="formula-variables">
                            <li>\(m\) is the number of AI models.</li>
                            <li>\(p_i\) is the normalized similarity score for model \(i\).</li>
                            <li>\(d_i\) is the distance metric measuring semantic difference of AI \(i\)'s response to the consolidated answer (e.g., cosine distance on text embeddings).</li>
                        </ul>

                        <h4>Derivation</h4>
                        <ol className="derivation-steps">
                            <li>
                                <strong>Calculate semantic distances.</strong>
                                <p>Calculate semantic distances \(d_i\) for each AI response compared to the consolidated answer, using metrics like cosine distance on text embeddings.</p>
                            </li>
                            <li>
                                <strong>Convert distances to similarity scores.</strong>
                                <p>Convert distances to similarity scores via</p>
                                <div className="formula-box">
                                    $$s_i = \exp(-d_i)$$
                                </div>
                            </li>
                            <li>
                                <strong>Normalize similarity scores for probability interpretation.</strong>
                                <p>Normalize similarity scores to obtain</p>
                                <div className="formula-box" dangerouslySetInnerHTML={{ __html: '$$p_i = \\frac{s_i}{\\sum_{j=1}^{m} s_j}$$' }}></div>
                            </li>
                            <li>
                                <strong>Compute entropy to measure agreement.</strong>
                                <p>Compute the entropy</p>
                                <div className="formula-box" dangerouslySetInnerHTML={{ __html: '$$\\mathrm{CACE}_q = - \\sum_{i=1}^{m} p_i \\log(p_i)$$' }}></div>
                                <p>which measures the level of agreement among AI models.</p>
                            </li>
                        </ol>

                        <div className="metric-explanation">
                            <p>
                                <strong>When the CACE score is low,</strong> it shows that the AI models mostly agree with each other in their answers.
                            </p>
                            <p>
                                <strong>When the CACE score is high,</strong> it means the AI models have very different or conflicting answers, signaling less agreement or more disagreement among them.
                            </p>
                        </div>

                        <CACEGraph />
                    </div>

                    <div className="metric-card">
                        <h3>Formula 3: Custom Balanced-Usage Loss</h3>
                        <p>
                            The Custom Balanced-Usage Loss function encourages neural network weights to achieve a balanced distribution,
                            promoting efficient weight utilization across neurons. This regularization technique helps prevent over-reliance
                            on specific weights while maintaining model expressiveness.
                        </p>

                        <h4>Loss Definition</h4>
                        <p>
                            For one data point (x, y) with prediction ŷ = f_W(x), the loss is:
                        </p>

                        <div className="formula-box" dangerouslySetInnerHTML={{ __html: '$$L(\\mathbf{W}; \\mathbf{x}, y) = (\\hat{y} - y)^2 + \\lambda \\sum_{j=1}^{m} (R_j - 1)^2$$' }}></div>

                        <p>where the balance ratio R<sub>j</sub> for each neuron j is defined as:</p>

                        <div className="formula-box" dangerouslySetInnerHTML={{ __html: '$$R_j = \\frac{S_j}{D_j} = \\frac{\\sum_{i=1}^{d} |W_{ij}|}{\\sqrt{\\sum_{i=1}^{d} W_{ij}^2 + \\varepsilon}}$$' }}></div>

                        <h4>Key Components</h4>
                        <ul className="formula-variables">
                            <li><strong>W</strong>: Weight matrix of the layer</li>
                            <li><strong>S<sub>j</sub></strong>: L1-norm (sum of absolute weights)</li>
                            <li><strong>Q<sub>j</sub></strong>: L2-norm squared (sum of weight squares)</li>
                            <li><strong>D<sub>j</sub></strong>: Stabilized L2 norm</li>
                            <li><strong>R<sub>j</sub></strong>: Balance ratio (optimal at 1)</li>
                            <li><strong>λ</strong>: Controls the penalty strength</li>
                            <li><strong>ε</strong>: Small stability constant</li>
                        </ul>

                        <div className="metric-explanation">
                            <p>
                                <strong>The penalty term (R<sub>j</sub> - 1)²</strong> encourages each neuron&apos;s weights to maintain a balance
                                between L1 and L2 norms. When R<sub>j</sub> = 1, the weights are optimally balanced, and the penalty is minimized.
                            </p>
                            <p>
                                <strong>The gradient</strong> of this loss with respect to each weight W<sub>ij</sub> combines the standard backpropagation
                                term with the penalty gradient, enabling efficient training via gradient descent.
                            </p>
                        </div>

                        <CustomLossGraph />
                    </div>
                </div>

                <div className="approach-section">
                    <div className="conclusion-card">
                        <h2>Our Goal</h2>
                        <p>
                            Overall, the platform aims to provide a <strong>robust, measurable, and interpretable</strong> way to assess AI model performance and trustworthiness across diverse inputs, empowering users to make informed decisions about AI reliability.
                        </p>
                    </div>
                </div>
            </div>
            <Footer />
        </>
    );
}

export default ApproachSolution;
