#!/usr/bin/env python3
"""
CLI tool to test the RAG system by asking questions.
"""

import sys
import argparse
import requests
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box

console = Console()


def ask_question(
    query: str,
    api_url: str = "http://localhost:8000",
    multi_query: bool = True,
    hyde: bool = False,
    top_k: int = None,  # None = auto-detect
    timeout: int = 2000
):
    """Ask a question to the RAG system."""
    
    endpoint = f"{api_url}/ask"
    
    payload = {
        "query": query,
        "multi_query": multi_query,
        "use_hyde": hyde,
    }
    
    # Only add top_k if explicitly set
    if top_k is not None:
        payload["top_k"] = top_k
    
    console.print(f"\n[cyan]❓ Kysymys:[/cyan] {query}")
    if top_k is None:
        console.print(f"[dim]API: {endpoint} (auto top_k)[/dim]\n")
    else:
        console.print(f"[dim]API: {endpoint} (top_k={top_k})[/dim]\n")
    
    try:
        with console.status("[bold cyan]Haetaan vastausta...", spinner="dots"):
            response = requests.post(
                endpoint,
                json=payload,
                timeout=timeout
            )
        
        response.raise_for_status()
        result = response.json()
        
        # Display answer
        console.print(Panel(
            Markdown(result['answer']),
            title="[bold green]💡 Vastaus[/bold green]",
            border_style="green",
            box=box.ROUNDED
        ))
        
        # Display sources
        if result.get('sources'):
            console.print("\n[bold cyan]📚 Lähteet:[/bold cyan]\n")
            
            sources_table = Table(box=box.SIMPLE)
            sources_table.add_column("#", style="dim", width=3)
            sources_table.add_column("Tiedosto", style="cyan")
            sources_table.add_column("Sijainti", style="yellow")
            sources_table.add_column("Katkelma", style="dim")
            
            for i, source in enumerate(result['sources'], 1):
                sources_table.add_row(
                    str(i),
                    source['file_name'][:40],
                    source.get('locator', 'N/A')[:20],
                    source.get('snippet', '')[:50] + "..."
                )
            
            console.print(sources_table)
            
            # Print clickable links
            console.print("\n[dim]🔗 Linkit:[/dim]")
            for i, source in enumerate(result['sources'], 1):
                console.print(f"  {i}. {source['link']}")
        
        # Display metadata
        console.print(f"\n[dim]⏱️  Vasteaika: {result.get('latency_ms', 0):.0f}ms[/dim]")
        
        return result
        
    except requests.exceptions.Timeout:
        console.print("[red]❌ Aikakatkaistu! Yritä pienemmällä top_k arvolla tai pidemmällä timeoutilla.[/red]")
        return None
    except requests.exceptions.ConnectionError:
        console.print(f"[red]❌ Ei yhteyttä API:in osoitteessa {api_url}[/red]")
        console.print("[yellow]Varmista että backend on käynnissä: docker-compose up -d[/yellow]")
        return None
    except requests.exceptions.HTTPError as e:
        console.print(f"[red]❌ HTTP virhe: {e}[/red]")
        if response.text:
            console.print(f"[dim]{response.text}[/dim]")
        return None
    except Exception as e:
        console.print(f"[red]❌ Virhe: {e}[/red]")
        return None


def search_documents(
    query: str,
    api_url: str = "http://localhost:8000",
    k: int = 20,
    timeout: int = 30
):
    """Search documents without answer generation."""
    
    endpoint = f"{api_url}/search"
    
    payload = {
        "query": query,
        "k": k
    }
    
    console.print(f"\n[cyan]🔍 Haku:[/cyan] {query}")
    console.print(f"[dim]API: {endpoint}[/dim]\n")
    
    try:
        with console.status("[bold cyan]Haetaan dokumentteja...", spinner="dots"):
            response = requests.post(
                endpoint,
                json=payload,
                timeout=timeout
            )
        
        response.raise_for_status()
        results = response.json()
        
        if not results:
            console.print("[yellow]Ei tuloksia.[/yellow]")
            return []
        
        console.print(f"[green]Löytyi {len(results)} tulosta:[/green]\n")
        
        # Display results
        for i, result in enumerate(results, 1):
            console.print(Panel(
                f"[bold]{result['file_name']}[/bold]\n"
                f"[dim]{result.get('locator', 'N/A')}[/dim]\n\n"
                f"{result['text'][:200]}...\n\n"
                f"[dim]Relevanssi: {result.get('score', 0):.3f}[/dim]",
                title=f"[cyan]#{i}[/cyan]",
                border_style="blue"
            ))
        
        return results
        
    except Exception as e:
        console.print(f"[red]❌ Virhe: {e}[/red]")
        return None


def iterative_rag(
    query: str,
    api_url: str = "http://localhost:8000",
    timeout: int = 120  # 2 minutes for iterative process
):
    """Perform iterative agentic RAG search."""
    
    endpoint = f"{api_url}/ask-iterative"
    
    payload = {
        "query": query,
        "multi_query": True
    }
    
    console.print(f"\n[bold cyan]🤖 Iteratiivinen Agentti:[/bold cyan] {query}")
    console.print(f"[dim]API: {endpoint}[/dim]\n")
    
    try:
        with console.status("[bold cyan]Agentti tutkii iteratiivisesti (voi kestää 30-120s)...", spinner="dots"):
            response = requests.post(
                endpoint,
                json=payload,
                timeout=timeout
            )
        
        response.raise_for_status()
        result = response.json()
        
        # Display iteration history
        console.print(Panel(
            f"[bold]Iteraatiot: {result['total_iterations']} | "
            f"Lähteet: {result['total_sources']} | "
            f"Luottamus: {result['final_confidence']:.0%}[/bold]",
            border_style="cyan"
        ))
        
        for iteration in result['iterations']:
            status = "✓" if iteration['confidence'] >= 0.85 else "↻"
            console.print(f"\n[bold yellow]{status} Kierros {iteration['iteration']}:[/bold yellow]")
            console.print(f"  Kysymys: {iteration['query']}")
            console.print(f"  Tulokset: {iteration['num_results']} lähdettä")
            console.print(f"  Luottamus: {iteration['confidence']:.0%}")
            console.print(f"  Arvio: {iteration['assessment']}")
            if iteration['missing_info']:
                console.print(f"  [dim]Puuttuu: {', '.join(iteration['missing_info'][:3])}[/dim]")
        
        # Display final answer
        console.print("\n")
        console.print(Panel(
            Markdown(result['answer']),
            title="[bold green]💡 Kattava Vastaus[/bold green]",
            border_style="green",
            box=box.ROUNDED
        ))
        
        # Display sources
        if result.get('sources'):
            console.print("\n[bold cyan]📚 Kaikki Lähteet:[/bold cyan]\n")
            
            sources_table = Table(box=box.SIMPLE)
            sources_table.add_column("#", style="dim", width=3)
            sources_table.add_column("Tiedosto", style="cyan")
            sources_table.add_column("Sijainti", style="yellow")
            sources_table.add_column("Katkelma", style="dim")
            
            for i, source in enumerate(result['sources'][:20], 1):  # Show first 20
                sources_table.add_row(
                    str(i),
                    source['file_name'][:40],
                    source.get('locator', 'N/A')[:20],
                    source.get('snippet', '')[:50] + "..."
                )
            
            console.print(sources_table)
            
            if len(result['sources']) > 20:
                console.print(f"\n[dim]... ja {len(result['sources']) - 20} muuta lähdettä[/dim]")
        
        console.print(f"\n[dim]⏱️  Kokonaisaika: {result.get('latency_ms', 0)/1000:.1f}s[/dim]")
        
        return result
        
    except requests.exceptions.Timeout:
        console.print("[red]❌ Aikakatkaistu! Iteratiivinen prosessi kestää liian kauan.[/red]")
        return None
    except requests.exceptions.ConnectionError:
        console.print(f"[red]❌ Ei yhteyttä API:in osoitteessa {api_url}[/red]")
        return None
    except Exception as e:
        console.print(f"[red]❌ Virhe: {e}[/red]")
        return None


def deep_research(
    query: str,
    api_url: str = "http://localhost:8000",
    timeout: int = 3000  # Longer timeout for research
):
    """Perform deep iterative research on a topic."""
    
    endpoint = f"{api_url}/research"
    
    payload = {
        "query": query,
        "multi_query": True
        # top_k will be auto-calculated if not provided
    }
    
    console.print(f"\n[bold cyan]🔬 Syvätutkimus:[/bold cyan] {query}")
    console.print(f"[dim]API: {endpoint}[/dim]\n")
    
    try:
        with console.status("[bold cyan]Tutkitaan dokumentteja iteratiivisesti...", spinner="dots"):
            response = requests.post(
                endpoint,
                json=payload,
                timeout=timeout
            )
        
        response.raise_for_status()
        result = response.json()
        
        # Display research steps
        console.print(Panel(
            f"[bold]Tutkimusvaiheet: {result['num_sub_questions']}[/bold]",
            border_style="cyan"
        ))
        
        for i, step in enumerate(result['research_steps'], 1):
            console.print(f"\n[bold yellow]Vaihe {i}:[/bold yellow] {step['question']}")
            answer_preview = step['answer'][:200]
            if len(step['answer']) > 200:
                answer_preview += "..."
            console.print(f"[dim]→ {answer_preview}[/dim]")
        
        # Display final synthesis
        console.print("\n")
        console.print(Panel(
            Markdown(result['answer']),
            title="[bold green]📊 Synteesivastaus[/bold green]",
            border_style="green",
            box=box.ROUNDED
        ))
        
        # Display sources
        if result.get('sources'):
            console.print("\n[bold cyan]📚 Lähteet:[/bold cyan]\n")
            
            sources_table = Table(box=box.SIMPLE)
            sources_table.add_column("#", style="dim", width=3)
            sources_table.add_column("Tiedosto", style="cyan")
            sources_table.add_column("Sijainti", style="yellow")
            sources_table.add_column("Katkelma", style="dim")
            
            for i, source in enumerate(result['sources'], 1):
                sources_table.add_row(
                    str(i),
                    source['file_name'][:40],
                    source.get('locator', 'N/A')[:20],
                    source.get('snippet', '')[:50] + "..."
                )
            
            console.print(sources_table)
            
            # Print clickable links
            console.print("\n[dim]🔗 Linkit:[/dim]")
            for i, source in enumerate(result['sources'], 1):
                if source.get('link'):
                    console.print(f"  {i}. {source['link']}")
        
        console.print(f"\n[dim]⏱️  Tutkimusaika: {result.get('latency_ms', 0)/1000:.1f}s[/dim]")
        
        return result
        
    except requests.exceptions.Timeout:
        console.print("[red]❌ Aikakatkaistu! Tutkimus kestää liian kauan.[/red]")
        console.print("[yellow]Yritä yksinkertaisempaa kysymystä tai pidennä timeoutia.[/yellow]")
        return None
    except requests.exceptions.ConnectionError:
        console.print(f"[red]❌ Ei yhteyttä API:in osoitteessa {api_url}[/red]")
        console.print("[yellow]Varmista että backend on käynnissä: docker-compose up -d[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]❌ Virhe: {e}[/red]")
        return None


def check_health(api_url: str = "http://localhost:8000"):
    """Check API health."""
    
    try:
        response = requests.get(f"{api_url}/healthz", timeout=5)
        response.raise_for_status()
        
        console.print(f"[green]✅ API toimii: {api_url}[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ API ei vastaa: {api_url}[/red]")
        console.print(f"[yellow]Käynnistä backend: docker-compose up -d[/yellow]")
        return False


def interactive_mode(api_url: str = "http://localhost:8000"):
    """Interactive question-answer session."""
    
    console.print(Panel(
        "[bold cyan]🤖 RAG Testi - Interaktiivinen tila[/bold cyan]\n\n"
        "Kirjoita kysymyksiä ja paina Enter.\n"
        "Komennot:\n"
        "  /iterative <kysymys> - Iteratiivinen agentti (hakee kunnes tyytyväinen)\n"
        "  /search <kysymys> - Hae dokumentteja ilman vastausta\n"
        "  /research <aihe> - Syvätutkimus aiheesta (iteratiivinen analyysi)\n"
        "  /multi - Vaihda multi-query päälle/pois\n"
        "  /hyde - Vaihda HyDE päälle/pois\n"
        "  /topk <n> - Aseta top_k arvo (tai 'auto' automaattiseen)\n"
        "  /quit - Lopeta\n",
        border_style="cyan"
    ))
    
    # Check health first
    if not check_health(api_url):
        return
    
    multi_query = True
    hyde = False
    top_k = None  # None = auto-detect
    
    console.print(f"\n[dim]Asetukset: multi_query={multi_query}, hyde={hyde}, top_k={'auto' if top_k is None else top_k}[/dim]\n")
    
    while True:
        try:
            query = console.input("\n[bold cyan]❓ Kysymys:[/bold cyan] ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.startswith('/'):
                cmd_parts = query.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                
                if cmd == '/quit':
                    console.print("[yellow]Näkemiin! 👋[/yellow]")
                    break
                
                elif cmd == '/multi':
                    multi_query = not multi_query
                    console.print(f"[green]Multi-query: {multi_query}[/green]")
                    continue
                
                elif cmd == '/hyde':
                    hyde = not hyde
                    console.print(f"[green]HyDE: {hyde}[/green]")
                    continue
                
                elif cmd == '/topk':
                    if len(cmd_parts) > 1:
                        arg = cmd_parts[1].lower()
                        if arg == 'auto':
                            top_k = None
                            console.print(f"[green]Top-K: auto (dynaaminen)[/green]")
                        else:
                            try:
                                top_k = int(arg)
                                console.print(f"[green]Top-K: {top_k}[/green]")
                            except ValueError:
                                console.print("[red]Virheellinen arvo. Käytä numeroa tai 'auto'[/red]")
                    else:
                        console.print(f"[yellow]Nykyinen top_k: {'auto' if top_k is None else top_k}[/yellow]")
                    continue
                
                elif cmd == '/search':
                    if len(cmd_parts) > 1:
                        search_documents(cmd_parts[1], api_url)
                    else:
                        console.print("[yellow]Anna hakukysely: /search <kysymys>[/yellow]")
                    continue
                
                elif cmd == '/iterative':
                    if len(cmd_parts) > 1:
                        iterative_rag(cmd_parts[1], api_url)
                    else:
                        console.print("[yellow]Anna kysymys: /iterative <kysymys>[/yellow]")
                    continue
                
                elif cmd == '/research':
                    if len(cmd_parts) > 1:
                        deep_research(cmd_parts[1], api_url)
                    else:
                        console.print("[yellow]Anna tutkimusaihe: /research <aihe>[/yellow]")
                    continue
                
                else:
                    console.print(f"[red]Tuntematon komento: {cmd}[/red]")
                    continue
            
            # Ask question
            ask_question(query, api_url, multi_query, hyde, top_k)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Näkemiin! 👋[/yellow]")
            break
        except EOFError:
            console.print("\n[yellow]Näkemiin! 👋[/yellow]")
            break


def main():
    parser = argparse.ArgumentParser(
        description='Test RAG system via CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esimerkkejä:
  # Interaktiivinen tila
  python scripts/test_ask.py

  # Yksittäinen kysymys
  python scripts/test_ask.py -q "Mikä on HOT:n puheenjohtaja?"

  # Iteratiivinen agentti (hakee kunnes tyytyväinen)
  python scripts/test_ask.py --iterative "etsi kaikki tanssikerhon puheenjohtajat"

  # Syvätutkimus (iteratiivinen analyysi)
  python scripts/test_ask.py --research "HOT:n hallinto ja organisaatio"

  # Haku ilman vastausta
  python scripts/test_ask.py --search "puheenjohtaja"

  # Käytä HyDE:ä
  python scripts/test_ask.py -q "projektin aikataulu" --hyde

  # Muuta top-k arvoa
  python scripts/test_ask.py -q "hallituksen kokoukset" --top-k 15

  # Tarkista API:n tila
  python scripts/test_ask.py --health
        """
    )
    
    parser.add_argument(
        '-q', '--query',
        type=str,
        help='Kysymys (jos ei annettu, käynnistyy interaktiivinen tila)'
    )
    
    parser.add_argument(
        '--iterative',
        type=str,
        help='Iteratiivinen agentti (hakee kunnes tyytyväinen, max 5 kierrosta)'
    )
    
    parser.add_argument(
        '--research',
        type=str,
        help='Syvätutkimus aiheesta (iteratiivinen, kattava analyysi)'
    )
    
    parser.add_argument(
        '--search',
        type=str,
        help='Hae dokumentteja ilman vastauksen generointia'
    )
    
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8000',
        help='API:n URL (oletus: http://localhost:8000)'
    )
    
    parser.add_argument(
        '--multi-query',
        action='store_true',
        default=True,
        help='Käytä multi-query laajennusta (oletus: True)'
    )
    
    parser.add_argument(
        '--no-multi-query',
        action='store_false',
        dest='multi_query',
        help='Älä käytä multi-query laajennusta'
    )
    
    parser.add_argument(
        '--hyde',
        action='store_true',
        help='Käytä HyDE (Hypothetical Document Embeddings)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=8,
        help='Montako lähdettä palautetaan (oletus: 8)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='HTTP timeout sekunneissa (oletus: 60)'
    )
    
    parser.add_argument(
        '--health',
        action='store_true',
        help='Tarkista API:n tila ja lopeta'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Tulosta vastaus JSON-muodossa'
    )
    
    args = parser.parse_args()
    
    # Health check mode
    if args.health:
        check_health(args.api_url)
        return
    
    # Iterative agentic RAG mode
    if args.iterative:
        result = iterative_rag(
            args.iterative,
            api_url=args.api_url,
            timeout=args.timeout * 2  # Double timeout for iterative
        )
        
        if args.json and result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    
    # Deep research mode
    if args.research:
        result = deep_research(
            args.research,
            api_url=args.api_url,
            timeout=args.timeout
        )
        
        if args.json and result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    
    # Search mode
    if args.search:
        results = search_documents(
            args.search,
            api_url=args.api_url,
            k=args.top_k,
            timeout=args.timeout
        )
        
        if args.json and results:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        return
    
    # Single question mode
    if args.query:
        result = ask_question(
            args.query,
            api_url=args.api_url,
            multi_query=args.multi_query,
            hyde=args.hyde,
            top_k=args.top_k,
            timeout=args.timeout
        )
        
        if args.json and result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    
    # Interactive mode (default)
    interactive_mode(args.api_url)


if __name__ == "__main__":
    main()
