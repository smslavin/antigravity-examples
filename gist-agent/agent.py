import os
import datetime
import requests
import arxiv
import PyPDF2
import io
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

KEYWORDS = [
    "large language models",
    "LLM",
    "reranking information retrieval",
    "embeddings",
    "vector representations",
    "transformer neural networks",
    "natural language processing",
    "NLP"
]

class GistAgent:
    def __init__(self):
        self.client = arxiv.Client()
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def search_arxiv(self):
        print("Searching arXiv...")
        query = " OR ".join([f'"{k}"' for k in KEYWORDS])
        search = arxiv.Search(
            query=query,
            max_results=100,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        results = []
        for result in self.client.results(search):
            results.append(result)
        return results

    def filter_and_deduplicate(self, papers):
        print("Filtering and deduplicating...")
        filtered = []
        seen_titles = set()
        
        # Calculate 24 hours ago
        yesterday = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
        
        for paper in papers:
            # Check date
            if paper.published < yesterday:
                continue
                
            # Deduplicate
            if paper.title in seen_titles:
                continue
            
            seen_titles.add(paper.title)
            filtered.append(paper)
            
        return filtered

    def rerank_papers(self, papers):
        print("Reranking...")
        # Simple heuristic: count keyword occurrences in title and abstract
        scored_papers = []
        for paper in papers:
            score = 0
            text = (paper.title + " " + paper.summary).lower()
            for keyword in KEYWORDS:
                if keyword.lower() in text:
                    score += 1
            scored_papers.append((score, paper))
        
        # Sort by score desc
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        return [p[1] for p in scored_papers[:10]]

    def download_pdf(self, paper):
        print(f"Downloading PDF for: {paper.title}")
        try:
            response = requests.get(paper.pdf_url)
            response.raise_for_status()
            return io.BytesIO(response.content)
        except Exception as e:
            print(f"Failed to download PDF: {e}")
            return None

    def extract_text_from_pdf(self, pdf_stream):
        if not pdf_stream:
            return ""
        try:
            reader = PyPDF2.PdfReader(pdf_stream)
            text = ""
            # Limit to first 5 pages to avoid huge context
            for i in range(min(len(reader.pages), 5)):
                text += reader.pages[i].extract_text()
            return text
        except Exception as e:
            print(f"Failed to extract text: {e}")
            return ""

    def analyze_paper(self, paper, pdf_text):
        print(f"Analyzing: {paper.title}")
        prompt = f"""
        Analyze the following academic paper.
        Title: {paper.title}
        Abstract: {paper.summary}
        Partial Content: {pdf_text[:10000]}...

        Please provide a concise summary, identify key trends, insights, implications for future research, and any research gaps mentioned or apparent.
        Format the output as Markdown.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Failed to analyze paper: {e}")
            return "Analysis failed."

    def generate_daily_update(self, analyzed_papers):
        print("Generating daily update...")
        
        # Generate overall trends
        all_summaries = "\n\n".join([a['analysis'] for a in analyzed_papers])
        trends_prompt = f"""
        Based on the following summaries of top papers from the last 24 hours, identify the overarching trends and insights in the field of LLMs and NLP.
        
        Summaries:
        {all_summaries}
        
        Output as a Markdown section titled "## Daily Trends and Insights".
        """
        try:
            trends_response = self.model.generate_content(trends_prompt)
            trends_section = trends_response.text
        except Exception as e:
            trends_section = "Could not generate overall trends."

        content = f"# arXiv Daily Update - {datetime.date.today()}\n\n"
        content += trends_section + "\n\n"
        content += "## Top Papers\n\n"
        
        for item in analyzed_papers:
            paper = item['paper']
            analysis = item['analysis']
            content += f"### [{paper.title}]({paper.entry_id})\n"
            content += f"**Published:** {paper.published.strftime('%Y-%m-%d')}\n\n"
            content += f"**Abstract:** {paper.summary}\n\n"
            content += f"**Analysis:**\n{analysis}\n\n"
            content += "---\n\n"
            
        return content

    def run(self):
        papers = self.search_arxiv()
        print(f"Found {len(papers)} papers.")
        
        filtered = self.filter_and_deduplicate(papers)
        print(f"Filtered to {len(filtered)} papers from last 24h.")
        
        top_papers = self.rerank_papers(filtered)
        print(f"Selected top {len(top_papers)} papers.")
        
        analyzed_data = []
        for paper in top_papers:
            pdf_stream = self.download_pdf(paper)
            pdf_text = self.extract_text_from_pdf(pdf_stream)
            analysis = self.analyze_paper(paper, pdf_text)
            analyzed_data.append({
                'paper': paper,
                'analysis': analysis
            })
            
        markdown_content = self.generate_daily_update(analyzed_data)
        
        with open("arxiv-daily-update.md", "w") as f:
            f.write(markdown_content)
        print("Done. Saved to arxiv-daily-update.md")

if __name__ == "__main__":
    agent = GistAgent()
    agent.run()
