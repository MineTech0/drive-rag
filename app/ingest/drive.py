"""Google Drive API integration for file discovery and download."""
import os
import hashlib
from typing import List, Dict, Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging

logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


class DriveClient:
    """Client for interacting with Google Drive API."""
    
    def __init__(self, credentials_path: str):
        """Initialize Drive client with service account credentials."""
        self.credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES
        )
        self.service = build('drive', 'v3', credentials=self.credentials)
    
    def list_files_recursive(
        self, 
        folder_id: str, 
        mime_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Recursively list all files in a folder and its subfolders.
        
        Args:
            folder_id: Google Drive folder ID
            mime_types: List of MIME types to filter (default: PDF and Google Docs)
            
        Returns:
            List of file metadata dictionaries
        """
        if mime_types is None:
            mime_types = [
                'application/pdf',
                'application/vnd.google-apps.document'
            ]
        
        all_files = []
        folders_to_process = [(folder_id, "")]
        
        while folders_to_process:
            current_folder_id, current_path = folders_to_process.pop(0)
            
            try:
                # Get folder name
                folder_info = self.service.files().get(
                    fileId=current_folder_id,
                    fields='name'
                ).execute()
                folder_name = folder_info.get('name', '')
                
                # Update path
                if current_path:
                    full_path = f"{current_path}/{folder_name}"
                else:
                    full_path = folder_name
                
                # List all items in current folder
                page_token = None
                while True:
                    query = f"'{current_folder_id}' in parents and trashed=false"
                    
                    response = self.service.files().list(
                        q=query,
                        spaces='drive',
                        fields='nextPageToken, files(id, name, mimeType, modifiedTime, '
                               'webViewLink, version, size)',
                        pageToken=page_token,
                        pageSize=100
                    ).execute()
                    
                    items = response.get('files', [])
                    
                    for item in items:
                        mime_type = item.get('mimeType', '')
                        
                        # If it's a folder, add to processing queue
                        if mime_type == 'application/vnd.google-apps.folder':
                            folders_to_process.append((item['id'], full_path))
                        
                        # If it's a target file type, add to results
                        elif mime_type in mime_types:
                            file_metadata = {
                                'file_id': item['id'],
                                'name': item['name'],
                                'path': full_path,
                                'mime_type': mime_type,
                                'modified_time': item.get('modifiedTime'),
                                'drive_link': item.get('webViewLink', ''),
                                'revision': item.get('version', ''),
                                'size': item.get('size', 0)
                            }
                            all_files.append(file_metadata)
                            logger.info(f"Found file: {file_metadata['name']} in {full_path}")
                    
                    page_token = response.get('nextPageToken')
                    if not page_token:
                        break
                        
            except HttpError as error:
                logger.error(f"Error accessing folder {current_folder_id}: {error}")
                continue
        
        logger.info(f"Total files found: {len(all_files)}")
        return all_files
    
    def export_document(self, file_id: str, mime_type: str = 'text/plain') -> str:
        """
        Export a Google Docs document to specified format.
        
        Args:
            file_id: Google Drive file ID
            mime_type: Export MIME type (default: text/plain)
            
        Returns:
            Exported document content as string
        """
        try:
            request = self.service.files().export_media(
                fileId=file_id,
                mimeType=mime_type
            )
            content = request.execute()
            
            # Decode bytes to string
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')
            
            return content
            
        except HttpError as error:
            logger.error(f"Error exporting document {file_id}: {error}")
            raise
    
    def download_file(self, file_id: str) -> bytes:
        """
        Download a file's content (for PDFs and other binary files).
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            File content as bytes
        """
        try:
            request = self.service.files().get_media(fileId=file_id)
            content = request.execute()
            return content
            
        except HttpError as error:
            logger.error(f"Error downloading file {file_id}: {error}")
            raise
    
    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute SHA256 hash of content for deduplication."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
