import httpx
import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
API_URL = "http://localhost:8000/search"

def main():
    console.print(Panel.fit("🔍 Semantic Code Search Engine CLI 🔍", style="bold blue"))
    console.print("Type your natural language query, or 'exit' to quit.\n")

    while True:
        try:
            query = Prompt.ask("[bold green]Query[/bold green]")
            if query.lower() in ('exit', 'quit'):
                break

            if not query.strip():
                continue

            results = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Searching...", total=None)

                try:
                    response = httpx.post(API_URL, json={"query": query, "top_k": 3}, timeout=10.0)
                    response.raise_for_status()
                    results = response.json().get("results", [])
                except httpx.RequestError as exc:
                    console.print(f"[bold red]An error occurred while requesting {exc.request.url!r}. Is the FastAPI server running?[/bold red]")
                    continue
                except httpx.HTTPStatusError as exc:
                    console.print(f"[bold red]Error response {exc.response.status_code} while requesting {exc.request.url!r}.[/bold red]")
                    continue

            if not results:
                console.print("[yellow]No results found.[/yellow]\n")
                continue

            console.print(f"\n[bold]Top Results for: '{query}'[/bold]\n")

            for rank, res in enumerate(results, 1):
                score = res["score"]
                desc = res["description"]
                code = res["code"]

                header = Text()
                header.append(f"Rank {rank} ", style="bold magenta")
                header.append(f"(Score: {score:.4f})\n", style="cyan")
                header.append(f"{desc}", style="italic")

                syntax = Syntax(code, "python", theme="monokai", line_numbers=False)

                panel = Panel(
                    syntax,
                    title=header,
                    title_align="left",
                    border_style="green"
                )
                console.print(panel)
                console.print()

        except KeyboardInterrupt:
            break

    console.print("\n[bold blue]Goodbye![/bold blue]")

if __name__ == "__main__":
    main()