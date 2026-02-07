# MISAI (Misinformation AI Shield)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)
![Gemini](https://img.shields.io/badge/Google-Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**AI Agents for Trustworthy Fact-Checking: A Reliability Evaluation Approach for Misinformation Detection**


---

## 📖 Table of Contents
- [Abstract](#-abstract)
- [Application Modules](#-application-modules)
    - [Intro & Approach Dashboard](#1-intro--approach-dashboard)
    - [MisBot: The AI Truth Sentinel](#2-misbot-the-ai-truth-sentinel)
    - [TestImage: Deepfake Forensics](#3-testimage-deepfake-forensics)
    - [TestVideo: Synthetic Media Scanner](#4-testvideo-synthetic-media-scanner)
    - [TestAI: Model Benchmarking](#5-testai-model-benchmarking)
- [Methodology & Formulas](#-methodology--formulas)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Getting Started](#-getting-started)
- [References](#-references)

---

## 📄 Abstract
Misinformation spreads quickly on the internet, and AI agents that verify news sometimes produce incorrect or misleading results. **MISAI** introduces an AI agent designed to evaluate the reliability of other AI agents used for fact-checking news and information. The evaluation agent tests these fact-checking agents using various scenarios, such as identifying false or fabricated facts, verifying citations, and resisting misleading or biased questions. By applying clear metrics and formulas, including **Reliability Rate** and **Hallucination Rate**, the system calculates an overall score to measure each agent's trustworthiness. This meta-verification process helps ensure that only dependable AI agents provide verified information to the public.

---

## 🚀 Application Modules

### 1. Intro & Approach Dashboard
**The Gateway to Truth**
The application opens with an immersive **Intro Screen** featuring 3D visualizations powered by Spline, setting the stage for a premium user experience.
- **Approach Solution**: A dedicated dashboard that visualizes the core mathematical models driving MISAI.
- **Real-Time Graphs**: Interactive charts displaying **CARS** (Reliability), **CACE** (Consistency), and **Custom Loss** metrics, allowing users to understand the theoretical backbone of the system before diving into tools.

### 2. MisBot: The AI Truth Sentinel
**Real-Time Multi-Model Fact Checking**
MisBot is not just a chatbot; it's an intelligent consensus engine.
- **Multi-Model Aggregation**: Simultaneously queries **Gemini**, **Groq** (Llama 3, Mixtral), and **OpenAI** (GPT-4o).
- **Live Grounding**: Cross-references every claim with live data from **Google Search (SERP)** and **Wikipedia** to prevent hallucinations.
- **Consolidated Insights**: Synthesizes a single, most reliable answer, highlighting the "Best Model" based on the **CARS** score.

### 3. TestImage: Deepfake Forensics
**Pixel-Level Manipulation Detection**
A powerful tool designed to uncover the invisible traces of editing and AI generation.
- **Analysis Workflow**: Upload an image to scan for artifacts consistent with GANs or diffusion models.
- **Detailed Reporting**: Provides a confidence score and highlights specific regions of interest (ROI) that show signs of tampering.

### 4. TestVideo: Synthetic Media Scanner
**Frame-by-Frame Authenticity Verification**
Combats the rising threat of deepfake videos.
- **Temporal Analysis**: Scans video content for inconsistencies in facial expressions, lighting, and audio-visual synchronization.
- **Deepfake Detection**: Identifies swapped faces, synthetic audio tracks, and spliced frames with high precision.

### 5. TestAI: Model Benchmarking
**The Meta-Evaluation Arena**
A dedicated interface for benchmarking different AI models against known misinformation scenarios.
- **Performance Metrics**: Visualizes hallucination tendencies, factual accuracy, and refusal rates for various models.
- **Leaderboard**: Ranks models based on their **CARS** and **CACE** scores, helping users choose the most trustworthy engine.

---

## 🧮 Methodology & Formulas

### 1. Composite AI Reliability Score (CARS)
The **CARS** metric evaluates an AI model $i$ across $n$ queries as a weighted combination of four components: Factual Accuracy ($A_i$), Reasoning Depth ($R_i$), Consistency ($C_i$), and Source Verification Confidence ($V_i$).

$$
\text{CARS}_i = A_i^{\alpha} R_i^{\beta} C_i^{\gamma} V_i^{\delta}
$$

Where $\alpha, \beta, \gamma, \delta \in [0,1]$ are weights summing to 1.

**Component Definitions:**
*   **Factual Accuracy ($A_i$)**: Ratio of correct facts to total facts.
    $$ A_i = \frac{1}{n} \sum_{k=1}^n \frac{\text{correct facts}_k}{\text{total facts}_k} $$
*   **Reasoning Depth ($R_i$)**: Average semantic complexity of explanations.
    $$ R_i = \frac{1}{n} \sum_{k=1}^n \log(1 + \text{conceptual nodes}) $$
*   **Consistency ($C_i$)**: Statistical stability of answers across multiple queries.
    $$ C_i = \frac{1}{1 + \text{Var}(S)} $$
*   **Source Verification ($V_i$)**: Overlap ratio with validated sources.
    $$ V_i = \frac{\text{verified statements}}{\text{total statements}} $$

### 2. Cross-AI Consistency Entropy (CACE)
**CACE** measures the consensus level among different AI models. A low score indicates high agreement, while a high score signals disagreement or hallucination risks.

$$
\text{CACE} = -\sum_{i=1}^m p_i \log p_i
$$

Where $p_i$ is the softmax normalized similarity score derived from the semantic distance $d_i$ of AI $i$'s response to the consolidated answer:
$$ p_i = \frac{\exp(-d_i)}{\sum_{j=1}^m \exp(-d_j)} $$

### 3. Custom Loss Function
For the neural network component, we utilize a custom loss function that includes a balanced-usage penalty:

$$
L_W(x, y) = ||y - f(wx)||^2 + \lambda \sum_{j=1}^m \left| \frac{\sum_{i=1}^d |W_{ij}|}{\epsilon + \sqrt{\sum_{i=1}^d W_{ij}^2}} - 1 \right|
$$

*   **First term**: Normal squared prediction error.
*   **Second term**: Balanced-usage penalty for neuron weights.
*   $\lambda$: Strength of the penalty.

---

## 🏗 System Architecture
MISAI platform is built on a robust, modular architecture designed for scalability and real-time processing.

1.  **User Interface (Frontend)**: Built with **React** and **Vite**, featuring a responsive design and interactive visualizations using **Recharts** and **Spline**.
2.  **API Gateway (Backend)**: A high-performance **FastAPI** server handles requests, manages sessions, and orchestrates AI model interactions.
3.  **Intelligence Layer**:
    *   **LLM Aggregator**: Connects to Gemini, Groq, and OpenAI APIs.
    *   **Grounding Engine**: Fetches real-time data from SERP and Wikipedia.
    *   **Scoring Engine**: Computes CARS and CACE scores using NumPy.
4.  **Media Analysis Engine**: Specialized modules for image and video deepfake detection.

---

## 💻 Technology Stack

### Frontend
- **Framework**: [React](https://react.dev/) with [Vite](https://vitejs.dev/)
- **Visualization**: [Recharts](https://recharts.org/) for dynamic score graphs
- **3D Elements**: [Spline](https://spline.design/) (via `@splinetool/react-spline`)
- **Styling**: CSS3 with responsive design

### Backend
- **API Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **AI Integration**: Google Gemini, Groq, OpenAI
- **Utilities**: `httpx` (Async HTTP), `numpy` (Math), `python-dotenv` (Config)

---

## 🛠 Getting Started

### Prerequisites
- **Node.js** (v14+)
- **Python** (3.9+)

### Installation

#### 1. Backend Setup
```bash
cd Backend
pip install -r requirements.txt
```
Create a `.env` file in `Backend/` with your API keys:
```env
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
SERP_API_KEY=your_key_here
REALITY_API_KEY=your_key_here
REALITY_BASE=https://api.reality.com
```
Start the backend server:
```bash
uvicorn main:app --reload
```

#### 2. Frontend Setup
```bash
cd Frontend
npm install
```
Create a `.env` file in `Frontend/`:
```env
VITE_HOST_URL=http://localhost:8000
```
Start the development server:
```bash
npm run dev
```

---

## 🔮 Business Model & Future Work
The platform is designed for media organizations, government bodies, and social media platforms. It operates on a hybrid model including subscription tiers and API usage fees. Future extensions include multimodal misinformation evaluation, real-time monitoring, and blockchain-backed audit mechanisms.

---

## 📚 References
1. Z. Cui et al., "Toward Verifiable Misinformation Detection: A Multi-Tool LLM Agent," arXiv:2508.03092, 2025.
2. "Hallucination to Truth: A Review of Fact-Checking and Factuality in LLMs," arXiv:2508.03860, 2025.
3. "Synthetic Lies: Understanding AI-Generated Misinformation," CHI '23.
4. "Truth Sleuth & Trend Bender AI Agents to fact-check YouTube Videos," arXiv:2507.10577, 2025.
5. J. A. S. de Cerqueira et al., "Can We Trust AI Agents? A Case Study of an LLM-Based Ethical Review System," arXiv:2411.08881, 2024.

---
*Date: November 28, 2025*
