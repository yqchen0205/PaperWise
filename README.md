# PaperWise 🧠📄

Intelligent paper parsing and summarization with layered "onion-style" analysis.

PaperWise is a Python pipeline that transforms academic papers into structured, actionable summaries using:
- **MinerU** for high-quality PDF parsing
- **LLM (OpenAI-compatible)** for intelligent multi-layer summarization

## ✨ Features

- 📄 **PDF Parsing**: Extract clean markdown from academic papers using MinerU API
- 🧅 **Layered Summarization**: "Onion-style" five-layer analysis:
  1. **TL;DR** - One-line summary
  2. **Motivation & Gap** - Problem statement and research gap
  3. **Method & Mechanism** - Technical approach and innovations
  4. **Proof & Results** - Key findings and SOTA comparisons
  5. **Insights & Decision** - Limitations, future work, and practical implications
- 📊 **Evidence Enrichment**: Automatically embeds referenced figures/tables from the original paper
- 📈 **Token Tracking**: Monitors LLM usage across each summarization stage
- 🔍 **Quality Checks**: Validates output structure and style coverage
- ⚙️ **Configurable**: Environment-based configuration with sensible defaults

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/paperwise.git
cd paperwise
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Configure

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Required: MinerU API Token (get from https://mineru.net)
MINERU_API_TOKEN=your_mineru_token_here

# Required: LLM API Key (OpenAI, Moonshot, or compatible)
OPENAI_API_KEY=your_api_key_here

# Optional: Custom base URL for OpenAI-compatible APIs
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini

# Output directory
OUTPUT_DIR=outputs
```

**⚠️ Security Note**: Never commit your `.env` file! It contains sensitive API keys. The `.gitignore` is already configured to exclude it.

### 3. Run

Process a directory of PDFs:

```bash
paperwise papers/
```

Process a single PDF:

```bash
paperwise papers/my_paper.pdf --output-dir ./results
```

Process a remote PDF URL:

```bash
paperwise --pdf-url "https://arxiv.org/pdf/2401.12345.pdf"
```

## 📁 Output Structure

```
outputs/
├── parsed_markdown/     # Clean markdown extracted from PDFs
├── summaries/          # Generated summaries with embedded figures
└── metadata/           # JSON with processing stats and quality metrics
```

## ⚙️ Configuration Reference

### Required Environment Variables

| Variable | Description | How to Get |
|----------|-------------|------------|
| `MINERU_API_TOKEN` | MinerU PDF parsing service key | Register at [mineru.net](https://mineru.net) |
| `OPENAI_API_KEY` | LLM API key | From your LLM provider (OpenAI, Moonshot, etc.) |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API endpoint URL |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Model identifier |
| `OUTPUT_DIR` | `outputs` | Where to save results |
| `PROMPT_TEMPLATE_PATH` | `prompt.md` | Custom summarization template |
| `SUMMARY_STORY_PLANNING_ENABLED` | `true` | Enable narrative planning stage |
| `SUMMARY_REVIEW_ENABLED` | `true` | Enable coherence review stage |
| `SUMMARY_REWRITE_ENABLED` | `true` | Enable final rewrite stage |
| `SUMMARY_LAYERED_GENERATION_ENABLED` | `true` | Enable five-layer generation |

### MinerU Options

| Variable | Default | Description |
|----------|---------|-------------|
| `MINERU_BASE_URL` | `https://mineru.net` | MinerU API endpoint |
| `MINERU_TIMEOUT_SEC` | `60` | Request timeout |
| `MINERU_POLL_INTERVAL_SEC` | `2.0` | Status polling interval |
| `MINERU_MODEL_VERSION` | `vlm` | Parsing model version |
| `MINERU_ENABLE_FORMULA` | `true` | Enable formula extraction |
| `MINERU_ENABLE_TABLE` | `true` | Enable table extraction |
| `MINERU_IS_OCR` | `true` | Enable OCR for scanned PDFs |
| `MINERU_LANGUAGE` | `en` | Document language |

## 🧪 Development

Run tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=src/papersummarizer --cov-report=html
```

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [MinerU](https://github.com/opendatalab/MinerU) for excellent PDF parsing
- Inspired by academic reading workflows and the need for better paper comprehension
