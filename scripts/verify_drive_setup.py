"""Script to verify Google Drive service account setup.

Usage:
    python scripts/verify_drive_setup.py
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import json


def verify_setup():
    """Verify Google Drive service account configuration."""
    console = Console()
    
    console.print("\n[bold cyan]Google Drive Setup Verification[/bold cyan]\n")
    
    issues = []
    warnings = []
    
    # Check 1: Service account file exists
    console.print("[cyan]1. Checking service account file...[/cyan]")
    sa_path = settings.google_application_credentials
    
    if not os.path.exists(sa_path):
        issues.append(f"Service account file not found: {sa_path}")
        console.print(f"   [red]✗ File not found: {sa_path}[/red]")
    else:
        console.print(f"   [green]✓ File exists: {sa_path}[/green]")
        
        # Check 2: Valid JSON
        console.print("[cyan]2. Validating JSON format...[/cyan]")
        try:
            with open(sa_path, 'r') as f:
                sa_data = json.load(f)
            console.print("   [green]✓ Valid JSON format[/green]")
            
            # Check 3: Required fields
            console.print("[cyan]3. Checking required fields...[/cyan]")
            required_fields = [
                'type', 'project_id', 'private_key_id', 
                'private_key', 'client_email', 'client_id'
            ]
            
            missing_fields = [f for f in required_fields if f not in sa_data]
            if missing_fields:
                issues.append(f"Missing fields in SA file: {', '.join(missing_fields)}")
                console.print(f"   [red]✗ Missing fields: {', '.join(missing_fields)}[/red]")
            else:
                console.print("   [green]✓ All required fields present[/green]")
                
                # Display service account info
                table = Table(show_header=False, box=None)
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="white")
                
                table.add_row("Type", sa_data.get('type', 'N/A'))
                table.add_row("Project ID", sa_data.get('project_id', 'N/A'))
                table.add_row("Client Email", sa_data.get('client_email', 'N/A'))
                table.add_row("Client ID", sa_data.get('client_id', 'N/A'))
                
                console.print("\n   [bold]Service Account Details:[/bold]")
                console.print(table)
                
                # Important reminder
                console.print("\n   [yellow]⚠ Important:[/yellow] Share your Drive folder with this email:")
                console.print(f"   [bold green]{sa_data.get('client_email')}[/bold green]")
                
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON in service account file: {e}")
            console.print(f"   [red]✗ Invalid JSON: {e}[/red]")
        except Exception as e:
            issues.append(f"Error reading service account file: {e}")
            console.print(f"   [red]✗ Error: {e}[/red]")
    
    # Check 4: .env configuration
    console.print("\n[cyan]4. Checking .env configuration...[/cyan]")
    
    if not settings.root_folder_id:
        warnings.append("ROOT_FOLDER_ID not set in .env")
        console.print("   [yellow]⚠ ROOT_FOLDER_ID not set[/yellow]")
    else:
        console.print(f"   [green]✓ ROOT_FOLDER_ID: {settings.root_folder_id}[/green]")
    
    if settings.embedding_backend == "vertex" and not settings.gcp_project_id:
        warnings.append("GCP_PROJECT_ID not set (needed for Vertex AI)")
        console.print("   [yellow]⚠ GCP_PROJECT_ID not set (needed for Vertex AI)[/yellow]")
    elif settings.gcp_project_id:
        console.print(f"   [green]✓ GCP_PROJECT_ID: {settings.gcp_project_id}[/green]")
    
    # Check 5: Try to initialize Drive client
    console.print("\n[cyan]5. Testing Drive API connection...[/cyan]")
    try:
        from app.ingest.drive import DriveClient
        drive_client = DriveClient(sa_path)
        console.print("   [green]✓ Drive client initialized successfully[/green]")
        
        # Try to list files if folder ID is set
        if settings.root_folder_id:
            console.print(f"\n[cyan]6. Testing access to folder {settings.root_folder_id}...[/cyan]")
            try:
                # Try to get folder info
                folder_info = drive_client.service.files().get(
                    fileId=settings.root_folder_id,
                    fields='name, id, mimeType'
                ).execute()
                
                console.print(f"   [green]✓ Successfully accessed folder: {folder_info.get('name')}[/green]")
                console.print(f"   [dim]Folder ID: {folder_info.get('id')}[/dim]")
                
            except Exception as e:
                if "404" in str(e):
                    issues.append("Folder not found or not shared with service account")
                    console.print(f"   [red]✗ Folder not found or not accessible[/red]")
                    console.print(f"   [yellow]→ Make sure the folder is shared with the service account email[/yellow]")
                elif "403" in str(e):
                    issues.append("Permission denied - folder not shared")
                    console.print(f"   [red]✗ Permission denied[/red]")
                    console.print(f"   [yellow]→ Share the folder with the service account as Viewer[/yellow]")
                else:
                    issues.append(f"Error accessing folder: {e}")
                    console.print(f"   [red]✗ Error: {e}[/red]")
        
    except Exception as e:
        issues.append(f"Failed to initialize Drive client: {e}")
        console.print(f"   [red]✗ Failed to initialize: {e}[/red]")
    
    # Summary
    console.print("\n" + "="*70)
    
    if not issues and not warnings:
        console.print(Panel(
            "[bold green]✓ All checks passed![/bold green]\n\n"
            "Your Google Drive setup is configured correctly.\n"
            "You can now run: [cyan]python scripts/list_drive_files.py --folder-id YOUR_ID[/cyan]",
            title="Success",
            border_style="green"
        ))
    elif issues:
        console.print(Panel(
            f"[bold red]✗ {len(issues)} issue(s) found:[/bold red]\n\n" +
            "\n".join(f"  • {issue}" for issue in issues),
            title="Issues Detected",
            border_style="red"
        ))
        console.print("\n[yellow]Please fix these issues before proceeding.[/yellow]\n")
    elif warnings:
        console.print(Panel(
            f"[bold yellow]⚠ {len(warnings)} warning(s):[/bold yellow]\n\n" +
            "\n".join(f"  • {warning}" for warning in warnings),
            title="Warnings",
            border_style="yellow"
        ))
        console.print("\n[cyan]Setup is mostly correct, but consider addressing warnings.[/cyan]\n")


if __name__ == "__main__":
    verify_setup()
