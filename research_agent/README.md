# Research Agent

A research paper summarization and search agent.

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/) (Miniconda or Anaconda)
- [Node.js](https://nodejs.org/) (v18 or higher)
- Google API Key (for Gemini LLM)


## Google API Key Configuration

To use the Gemini LLM, you need a Google API Key.

1.  Go to [Google AI Studio](https://aistudio.google.com/).
2.  Click on **Get API key**.
3.  Click **Create API key** (you can create it in a new or existing project).
4.  Copy the generated API key.
5.  You will use this key in the **Backend Setup** section below.

## Backend Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda create -n research_agent_env python=3.10
    conda activate research_agent_env
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the `backend` directory and add your Google API Key:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```

5.  **Start the Server:**
    ```bash
    python main.py
    ```
    The backend API will be available at [http://localhost:8003/docs](http://localhost:8003/docs).

## Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Start the Development Server:**
    ```bash
    npm run dev
    ```
    The frontend application will be available at [http://localhost:5173](http://localhost:5173).

## Troubleshooting

-   **Backend Crash**: If the backend crashes with a `PanicException`, ensure you have compatible versions of `sentence-transformers` and `chromadb`.
-   **Port Conflicts**: The backend is configured to run on port **8003** to avoid conflicts. The frontend proxy is configured accordingly.
