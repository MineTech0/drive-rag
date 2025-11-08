"""Google Docs parsing via Drive API export."""
import logging

logger = logging.getLogger(__name__)


def parse_google_doc(content: str) -> str:
    """
    Process exported Google Docs content.
    
    Google Docs are exported as text/plain via Drive API,
    so this function mainly performs cleaning and normalization.
    
    Args:
        content: Exported document content as string
        
    Returns:
        Cleaned text
    """
    try:
        # Remove excessive whitespace
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Strip trailing whitespace
            line = line.rstrip()
            # Keep the line if it has content or is a single empty line for paragraph breaks
            if line or (cleaned_lines and cleaned_lines[-1] != ''):
                cleaned_lines.append(line)
        
        # Join lines and normalize multiple blank lines to single blank line
        text = '\n'.join(cleaned_lines)
        
        # Remove excessive blank lines (more than 2 consecutive)
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error processing Google Doc content: {e}")
        return content


def parse_google_doc_html(html_content: str) -> str:
    """
    Parse Google Docs HTML export to extract text with structure.
    
    This is useful if we want to preserve heading hierarchy.
    
    Args:
        html_content: HTML content from Google Docs export
        
    Returns:
        Text with preserved structure
    """
    try:
        from html.parser import HTMLParser
        
        class DocsHTMLParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text_parts = []
                self.current_tag = None
            
            def handle_starttag(self, tag, attrs):
                self.current_tag = tag
                # Add markers for headings
                if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    self.text_parts.append(f"\n\n## {tag.upper()}: ")
            
            def handle_endtag(self, tag):
                if tag in ['p', 'div', 'li']:
                    self.text_parts.append('\n')
                elif tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    self.text_parts.append('\n')
                self.current_tag = None
            
            def handle_data(self, data):
                if data.strip():
                    self.text_parts.append(data.strip())
        
        parser = DocsHTMLParser()
        parser.feed(html_content)
        
        text = ' '.join(parser.text_parts)
        # Clean up excessive whitespace
        while '  ' in text:
            text = text.replace('  ', ' ')
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error parsing Google Doc HTML: {e}")
        # Fallback to plain text export
        return html_content
