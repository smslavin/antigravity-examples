from ingest import ingest_pdf
import os
import glob

def ingest_all():
    papers_dir = "../papers"
    pdf_files = glob.glob(os.path.join(papers_dir, "*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {papers_dir}")
    
    for pdf_file in pdf_files:
        print(f"Ingesting {pdf_file}...")
        ingest_pdf(pdf_file)

if __name__ == "__main__":
    ingest_all()
