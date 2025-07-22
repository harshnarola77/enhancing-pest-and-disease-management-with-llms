# Mini-PestMA: Enhancing Pest and Disease Management with LLMs

> A Multi-Agent System for Plant Health Analysis

## 📋 Project Overview

Mini-PestMA is a sophisticated multi-agent system designed to assist in the diagnosis of plant pests and diseases. By leveraging multiple Large Language Models (LLMs) running locally via Ollama, the system provides a robust, evidence-based analysis that goes beyond a single model's opinion.

The core of this project is a three-agent pipeline that mimics a professional diagnostic workflow:

1. **Diagnoser Agent** - Makes an initial assessment
2. **Validator Agent** - Critically reviews and challenges the diagnosis  
3. **Advisor Agent** - Synthesizes all information to provide practical, risk-adjusted recommendations

This project includes both a user-friendly web interface built with Streamlit and a command-line tool for development and evaluation.

## ✨ Key Features

- **Multi-Agent Architecture**: Utilizes three specialized AI agents for more accurate and reliable diagnosis
- **Local First**: Runs entirely on local hardware using Ollama, ensuring data privacy and offline capability
- **Image Analysis**: Can process uploaded images of plants alongside textual descriptions for comprehensive analysis
- **Robust JSON Parsing**: Features intelligent JSON extraction mechanism to handle varied LLM outputs and prevent crashes
- **Error Resilience**: Includes fallback mechanisms to ensure the system can recover from model or parsing errors gracefully
- **Dual Interfaces**:
  - Rich, interactive Streamlit Web UI for easy use
  - Command-Line Interface (CLI) for testing, evaluation, and backend development

## 🛠️ System Architecture

The Mini-PestMA system processes requests through a sequential pipeline of three distinct agents:

### 1. 🎯 Agent 1: The Critical Diagnoser
- **Model**: `mistral-small3.2:24b`
- **Role**: Acts as a forensic plant pathologist
- **Function**: Analyzes user descriptions and optional images to generate initial diagnosis, confidence scores, and key observational evidence
- **Output**: Structured JSON object

### 2. 🔍 Agent 2: The Skeptical Validator  
- **Model**: `gemma3:27b`
- **Role**: Acts as a quality reviewer
- **Function**: Receives JSON output from Agent 1 and challenges conclusions, checks for biases, assesses evidence quality, and provides confidence adjustments
- **Output**: Structured JSON object

### 3. 💡 Agent 3: The Conservative Advisor
- **Model**: `phi4:14b` 
- **Role**: Acts as a practical extension agent
- **Function**: Synthesizes findings from both previous agents into final, human-readable report with actionable advice, monitoring plan, and safety considerations
- **Output**: Human-readable report

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git
- [Ollama](https://ollama.ai/) installed and running on your system

### Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://df-git.informatik.uni-kl.de/digital-farming-seminar/enhancing-pest-and-disease-management-with-llms.git
   cd enhancing-pest-and-disease-management-with-llms
   ```

2. **Download the required LLM models**:
   
   Open your terminal and pull the three models required by the agents. This will take some time and disk space.
   
   ```bash
   ollama pull mistral-small3.2:24b
   ollama pull gemma3:27b
   ollama pull phi4:14b
   ```

3. **Install Python dependencies**:
   
   It is recommended to use a virtual environment.
   
   ```bash
   # Create and activate a virtual environment (optional but recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   
   # Install the required packages
   pip install -r requirements.txt
   ```

   **Note**: If a `requirements.txt` file is not present, create one with the following content:
   ```txt
   streamlit
   ollama
   Pillow
   ```

## 💻 Usage

You can run this project in two ways:

### 1. The Streamlit Web Interface (Recommended)

This is the main application for end-users. From your terminal, run:

```bash
streamlit run streamlit_optimized_app.py
```

This will open a new tab in your web browser with the Mini-PestMA interface.

### 2. The Command-Line Interface (for Development/Evaluation)

This script runs the core logic in the terminal, displays a detailed report, and can be used to run a full system evaluation.

```bash
python mini_pestma_main.py
```

## 📂 Project Structure

```
.
├── streamlit_optimized_app.py  # The Streamlit web application front-end
├── mini_pestma_main.py         # The core backend logic and CLI
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── LICENSE                     # MIT License file
```

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, please feel free to:

- Create a merge request
- Open an issue
- Submit feature requests

## 📄 License

This project is open source and available for anyone to use, modify, and distribute freely. No restrictions apply.

## 🙏 Acknowledgments

- Special thanks to [Vishal Sharbidar Mukunda] for his guidance and support
- This project was developed as part of the Digital Farming Seminar at [RPTU Kaiserslautern]

---

**Note**: Make sure Ollama is running before starting the application. The system requires all three models to be available locally for proper functionality.