# Gist Agent

A Python agent that scrapes arXiv for recent papers on Large Language Models (LLMs), reranking, and embeddings, and generates a daily update with summaries and insights.

## Features

- **Scrapes arXiv**: Searches for papers related to LLMs, NLP, and embeddings published in the last 24 hours.
- **Intelligent Filtering**: Deduplicates and reranks papers based on keyword relevance.
- **PDF Analysis**: Downloads PDFs and extracts text for deeper analysis.
- **AI Summarization**: Uses Google's Gemini model to summarize papers and identify trends, insights, and research gaps.
- **Gist Publication**: Automatically publishes the daily update as a public GitHub Gist.
- **Daily Update**: Generates a Markdown report (`arxiv-daily-update.md`) with the findings.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/smslavin/antigravity-examples.git
    cd antigravity-examples/gist-agent
    ```

2.  **Create a Conda environment**:
    ```bash
    conda create -n gistAgent python=3.11
    conda activate gistAgent
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment**:
    Create a `.env` file in the `gist-agent` directory with your Gemini API key and GitHub Token:
    ```bash
    cp .env.example .env
    # Edit .env and add your GEMINI_API_KEY and GITHUB_TOKEN
    ```
    *(Note: You need a Google Gemini API key and a GitHub Personal Access Token with `gist` scope)*

## Usage

Run the agent:

```bash
conda run -n gistAgent python agent.py
```

The agent will:
1.  Search arXiv.
2.  Filter and rerank papers.
3.  Download and analyze the top 10 papers.
4.  Generate `arxiv-daily-update.md`.

## Output

The output is:
- A Markdown file `arxiv-daily-update.md` containing the full report.
- A public GitHub Gist with the same content (URL printed to console).

The report contains:
- **Daily Trends and Insights**: A high-level summary of the day's research.
- **Top Papers**: Detailed summaries and analysis of the top 10 papers.
