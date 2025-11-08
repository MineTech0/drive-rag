#!/usr/bin/env python3
"""
Ingest documents from a CSV file listing specific files.
This allows selective ingestion by editing the CSV file.
"""

import csv
import sys
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from sqlalchemy import text as sql_text

# Suppress PDF parser warnings about malformed PDFs
logging.getLogger('pypdf').setLevel(logging.ERROR)
logging.getLogger('fitz').setLevel(logging.ERROR)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich import box

from app.config import settings
from app.database import get_db
from app.ingest.drive import DriveClient
from app.parse.pdf import parse_pdf
from app.parse.docs import parse_google_doc
from app.chunking.semantic import SemanticChunker
from app.index.pgvector import EmbeddingService, PgVectorIndexer

console = Console()


def clean_text_for_postgres(text: str) -> str:
    """Remove NUL bytes and other problematic characters for PostgreSQL."""
    if not text:
        return text
    
    # Remove NUL bytes
    text = text.replace('\x00', '')
    
    # Remove other control characters except newlines and tabs
    text = ''.join(char for char in text if char == '\n' or char == '\t' or ord(char) >= 32)
    
    return text.strip()


class IngestStats:
    def __init__(self):
        self.total = 0
        self.processed = 0
        self.skipped = 0
        self.errors = 0
        self.chunks_created = 0
        self.start_time = time.time()
        self.error_details: List[Dict] = []

    def add_error(self, file_name: str, error: str):
        self.errors += 1
        self.error_details.append({
            "file": file_name,
            "error": error,
            "time": datetime.now().isoformat()
        })

    def get_rate(self) -> float:
        elapsed = time.time() - self.start_time
        return self.processed / elapsed if elapsed > 0 else 0


def read_csv_files(csv_path: str) -> List[Dict]:
    """Read file list from CSV with automatic encoding detection."""
    files = []
    
    # Try different encodings (utf-8-sig handles BOM)
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(csv_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Strip whitespace and BOM from keys and values
                    cleaned_row = {}
                    for k, v in row.items():
                        if k:
                            # Remove BOM and whitespace from key
                            clean_key = k.strip().lstrip('\ufeff').strip()
                            # Strip whitespace from value
                            clean_value = v.strip() if isinstance(v, str) else v
                            cleaned_row[clean_key] = clean_value
                    
                    # Skip empty rows
                    if cleaned_row and any(v for v in cleaned_row.values()):
                        files.append(cleaned_row)
            console.print(f"[green]✓ CSV read successfully with {encoding} encoding[/green]")
            return files
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to read with {encoding}: {e}[/yellow]")
            continue
    
    # If all encodings fail, raise error
    raise ValueError(f"Could not read CSV file with any supported encoding: {encodings}")


def process_file(
    file_info: Dict,
    drive_client: DriveClient,
    chunker: SemanticChunker,
    indexer: PgVectorIndexer,
    stats: IngestStats,
    full_reindex: bool = False
) -> bool:
    """Process a single file from CSV."""
    file_id = file_info['file_id']
    file_name = file_info['name']
    mime_type = file_info['mime_type']
    drive_link = file_info['drive_link']
    path = file_info.get('path', '')
    modified_time = file_info.get('modified_time', '')

    try:
        # Check if already indexed
        with get_db() as db:
            if not full_reindex:
                existing = db.execute(
                    sql_text("SELECT content_sha256 FROM documents WHERE file_id = :file_id"),
                    {"file_id": file_id}
                ).fetchone()
                
                if existing:
                    stats.skipped += 1
                    return True

        # Parse document based on type
        if mime_type == 'application/pdf':
            # Download PDF
            pdf_bytes = drive_client.download_file(file_id)
            text = parse_pdf(pdf_bytes)
            metadata = {"source_type": "pdf"}
            
        elif mime_type == 'application/vnd.google-apps.document':
            # Export Google Doc
            text = drive_client.export_document(file_id, 'text/plain')
            metadata = {"source_type": "google_doc"}
            
        else:
            console.print(f"[yellow]Skipping unsupported type: {mime_type}[/yellow]")
            stats.skipped += 1
            return True

        if not text or not text.strip():
            console.print(f"[yellow]Empty content: {file_name}[/yellow]")
            stats.skipped += 1
            return True

        # Clean text for PostgreSQL (remove NUL bytes)
        text = clean_text_for_postgres(text)

        # Compute content hash
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

        # Check if content changed
        with get_db() as db:
            if not full_reindex:
                existing = db.execute(
                    sql_text("SELECT content_sha256 FROM documents WHERE file_id = :file_id"),
                    {"file_id": file_id}
                ).fetchone()
                
                if existing and existing[0] == content_hash:
                    stats.skipped += 1
                    return True

        # Chunk text
        chunks = chunker.chunk_text(
            text=text,
            metadata={
                **metadata,
                "file_id": file_id,
                "file_name": file_name,
                "path": path
            }
        )

        if not chunks:
            console.print(f"[yellow]No chunks created: {file_name}[/yellow]")
            stats.skipped += 1
            return True

        # Clean chunk texts for PostgreSQL
        for chunk in chunks:
            chunk['text'] = clean_text_for_postgres(chunk['text'])

        # Index document
        doc_id = indexer.upsert_document({
            'file_id': file_id,
            'name': file_name,
            'mime_type': mime_type,
            'content_sha256': content_hash,
            'drive_link': drive_link,
            'path': path,
            'modified_time': modified_time if modified_time else None
        })

        # Index chunks
        indexer.index_chunks(doc_id, chunks)
        
        stats.processed += 1
        stats.chunks_created += len(chunks)
        return True

    except Exception as e:
        stats.add_error(file_name, str(e))
        console.print(f"[red]Error processing {file_name}: {e}[/red]")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Ingest specific files from CSV list'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV file with file list (output from list_drive_files.py)'
    )
    parser.add_argument(
        '--full-reindex',
        action='store_true',
        help='Reprocess all files even if already indexed'
    )
    
    args = parser.parse_args()

    # Validate CSV file exists
    csv_path = Path(args.csv)
    if not csv_path.exists():
        console.print(f"[red]Error: CSV file not found: {args.csv}[/red]")
        sys.exit(1)

    # Read files from CSV
    console.print(f"\n[cyan]Reading file list from:[/cyan] {args.csv}")
    files = read_csv_files(str(csv_path))
    
    if not files:
        console.print("[red]No files found in CSV![/red]")
        sys.exit(1)

    # Validate CSV has required columns
    required_columns = ['file_id', 'name', 'mime_type', 'drive_link']
    if files:
        missing_columns = [col for col in required_columns if col not in files[0]]
        if missing_columns:
            console.print(f"[red]Error: CSV is missing required columns: {', '.join(missing_columns)}[/red]")
            console.print(f"[yellow]Found columns: {', '.join(files[0].keys())}[/yellow]")
            console.print(f"\n[cyan]Expected columns: {', '.join(required_columns)}[/cyan]")
            console.print("\n[yellow]Tip: Generate CSV with:[/yellow]")
            console.print("  python scripts/list_drive_files.py --folder-id YOUR_ID --format csv > files.csv")
            sys.exit(1)

    console.print(f"[green]Found {len(files)} files in CSV[/green]\n")

    # Initialize components
    stats = IngestStats()
    stats.total = len(files)

    try:
        console.print("[cyan]Initializing components...[/cyan]")
        drive_client = DriveClient(credentials_path=settings.google_application_credentials)
        chunker = SemanticChunker(
            max_tokens=settings.max_chunk_tokens,
            overlap_tokens=settings.chunk_overlap
        )
        
        # Convert SQLAlchemy URL to psycopg format
        db_url = settings.db_url.replace('postgresql+psycopg://', 'postgresql://')
        indexer = PgVectorIndexer(db_url=db_url)
        
        console.print("[green]✓ Components initialized[/green]\n")

        # Create progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                "[cyan]Processing files...",
                total=stats.total
            )

            # Process each file
            for file_info in files:
                file_name = file_info['name']
                progress.update(
                    task,
                    description=f"[cyan]Processing: {file_name[:50]}..."
                )
                
                process_file(
                    file_info=file_info,
                    drive_client=drive_client,
                    chunker=chunker,
                    indexer=indexer,
                    stats=stats,
                    full_reindex=args.full_reindex
                )
                
                progress.advance(task)

        # Display final statistics
        elapsed = time.time() - stats.start_time
        
        table = Table(title="Ingestion Complete", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        
        table.add_row("Total files", str(stats.total))
        table.add_row("Processed", str(stats.processed))
        table.add_row("Skipped", str(stats.skipped))
        table.add_row("Errors", str(stats.errors), style="red" if stats.errors > 0 else "green")
        table.add_row("Chunks created", str(stats.chunks_created))
        table.add_row("Elapsed time", f"{elapsed:.1f}s")
        table.add_row("Rate", f"{stats.get_rate():.1f} files/sec")
        
        console.print("\n")
        console.print(table)

        # Show errors if any
        if stats.error_details:
            console.print("\n[red]Errors:[/red]")
            error_table = Table(box=box.SIMPLE)
            error_table.add_column("File", style="yellow")
            error_table.add_column("Error", style="red")
            
            for error in stats.error_details[:10]:  # Show first 10 errors
                error_table.add_row(
                    error["file"][:50],
                    error["error"][:100]
                )
            
            console.print(error_table)
            
            if len(stats.error_details) > 10:
                console.print(f"\n[yellow]... and {len(stats.error_details) - 10} more errors[/yellow]")

        # Success summary
        if stats.errors == 0:
            console.print(Panel(
                f"[green]✓ Successfully processed {stats.processed} files![/green]",
                border_style="green"
            ))
        else:
            console.print(Panel(
                f"[yellow]⚠ Completed with {stats.errors} errors[/yellow]",
                border_style="yellow"
            ))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Partial progress saved.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
