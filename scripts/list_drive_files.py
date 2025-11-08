"""Script to list all indexable documents from a Google Drive folder.

Usage:
    python scripts/list_drive_files.py --folder-id YOUR_FOLDER_ID
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingest.drive import DriveClient
from app.config import settings
from rich.console import Console
from rich.table import Table
from rich.progress import Progress


def format_size(size_bytes):
    """Format file size in human readable format."""
    try:
        size = int(size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    except (ValueError, TypeError):
        return "Unknown"


def list_drive_files(folder_id: str, output_format: str = "table"):
    """
    List all indexable files from Google Drive folder.
    
    Args:
        folder_id: Google Drive folder ID
        output_format: Output format ('table', 'json', or 'csv')
    """
    console = Console()
    
    # Only print header for table format
    if output_format == "table":
        console.print(f"\n[bold cyan]Listing files from Google Drive folder:[/bold cyan] {folder_id}\n")
    
    # Initialize Drive client
    try:
        drive_client = DriveClient(settings.google_application_credentials)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to initialize Drive client: {e}")
        return
    
    # List files
    if output_format == "table":
        with Progress() as progress:
            task = progress.add_task("[cyan]Scanning Drive...", total=None)
            
            try:
                files = drive_client.list_files_recursive(folder_id)
                progress.update(task, completed=100, total=100)
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] Failed to list files: {e}")
                return
    else:
        # For JSON/CSV, don't show progress bar
        try:
            files = drive_client.list_files_recursive(folder_id)
        except Exception as e:
            if output_format == "table":
                console.print(f"[bold red]Error:[/bold red] Failed to list files: {e}")
            else:
                import sys
                sys.stderr.write(f"Error: Failed to list files: {e}\n")
            return
    
    if not files:
        if output_format == "table":
            console.print("[yellow]No indexable files found in the folder.[/yellow]")
        return
    
    # Calculate statistics
    total_files = len(files)
    pdf_count = sum(1 for f in files if f['mime_type'] == 'application/pdf')
    docs_count = sum(1 for f in files if f['mime_type'] == 'application/vnd.google-apps.document')
    total_size = sum(int(f.get('size', 0)) for f in files)
    
    # Display based on format
    if output_format == "table":
        # Create table
        table = Table(title=f"Indexable Files ({total_files} total)")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Name", style="green")
        table.add_column("Type", style="yellow", width=10)
        table.add_column("Path", style="blue")
        table.add_column("Size", style="magenta", width=10)
        table.add_column("Modified", style="white", width=12)
        
        for idx, file in enumerate(files, 1):
            file_type = "PDF" if "pdf" in file['mime_type'] else "Docs"
            size_str = format_size(file.get('size', 0))
            modified = file.get('modified_time', 'Unknown')[:10]  # Date only
            
            table.add_row(
                str(idx),
                file['name'],
                file_type,
                file['path'],
                size_str,
                modified
            )
        
        console.print(table)
        
        # Print statistics
        console.print(f"\n[bold]Statistics:[/bold]")
        console.print(f"  Total files: [cyan]{total_files}[/cyan]")
        console.print(f"  PDFs: [yellow]{pdf_count}[/yellow]")
        console.print(f"  Google Docs: [yellow]{docs_count}[/yellow]")
        console.print(f"  Total size: [magenta]{format_size(total_size)}[/magenta]")
        console.print(f"  Estimated chunks: [cyan]~{total_files * 10}[/cyan] (avg 10 per doc)")
        console.print()
        
    elif output_format == "json":
        import json
        print(json.dumps(files, indent=2, default=str))
        
    elif output_format == "csv":
        import csv
        import sys
        
        # Define fieldnames based on what we want to export
        fieldnames = ['name', 'mime_type', 'path', 'size', 'modified_time', 'file_id', 'drive_link', 'revision']
        
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=fieldnames,
            extrasaction='ignore',  # Ignore any extra fields not in fieldnames
            lineterminator='\n'  # Use only \n to avoid extra blank lines on Windows
        )
        writer.writeheader()
        writer.writerows(files)


def main():
    parser = argparse.ArgumentParser(
        description="List indexable documents from Google Drive folder"
    )
    parser.add_argument(
        "--folder-id",
        required=True,
        help="Google Drive folder ID"
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)"
    )
    
    args = parser.parse_args()
    
    list_drive_files(args.folder_id, args.format)


if __name__ == "__main__":
    main()
