import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ingest import ingest_pdf

class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            print(f"New PDF detected: {event.src_path}")
            # Wait a bit for file copy to finish
            time.sleep(1)
            ingest_pdf(event.src_path)

def start_watcher(path_to_watch):
    print(f"Watching {path_to_watch} for new PDFs...")
    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    # Assuming run from backend/ directory
    papers_dir = "../papers"
    if not os.path.exists(papers_dir):
        os.makedirs(papers_dir)
    start_watcher(papers_dir)
