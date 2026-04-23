#!/usr/bin/env python3
"""
Ollama Code Generator
A CLI tool to generate code snippets using a local Ollama instance,
with directory indexing (vault-style memory) for contextual retrieval.
"""

import argparse
import fnmatch
import json
import os
import re
import sys
import urllib.request
import urllib.error


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
DEFAULT_MODEL = "deepseek-v3.1:671b-cloud"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
EMBED_BATCH_SIZE = 32

SKIP_DIRS = {
    "node_modules",
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "dist",
    "build",
    "target",
    "out",
    ".next",
    "coverage",
    "site-packages",
    "eggs",
}

SKIP_FILES = {
    ".DS_Store",
    "Thumbs.db",
    ".env",
    ".env.local",
}

TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
    ".rst",
    ".txt",
    ".sh",
    ".zsh",
    ".bash",
    ".fish",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rs",
    ".go",
    ".java",
    ".kt",
    ".scala",
    ".rb",
    ".php",
    ".cs",
    ".swift",
    ".r",
    ".m",
    ".mm",
    ".sql",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".html",
    ".htm",
    ".xml",
    ".svg",
    ".dockerfile",
    ".makefile",
    ".cmake",
    ".gradle",
    ".ini",
    ".cfg",
    ".conf",
    ".properties",
    ".lock",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".env",
    ".env.local",
}

PERSONAS = {
    "default": (
        "You are a helpful coding assistant. "
        "Respond with clean, well-commented code snippets. "
        "Only output the code unless explanation is explicitly asked for."
    ),
    "sarcastic": (
        "You are a grizzled senior developer who has seen too much. "
        "You write correct, efficient code, but you pepper responses with dry, "
        "sarcastic commentary about the state of modern software."
    ),
    "teacher": (
        "You are a patient programming mentor. "
        "Write code with extensive inline comments explaining every decision, "
        "then briefly summarize the key concepts at the end."
    ),
    "minimalist": (
        "You are a minimalist hacker. "
        "Write the shortest, most elegant solution possible. "
        "No comments, no fluff, no markdown fences—just raw code."
    ),
    "rust-evangelist": (
        "You are an enthusiastic Rust advocate. "
        "Whatever language the user asks for, you first explain why Rust would be "
        "better, then grudgingly provide the requested code."
    ),
}


def _is_text_file(path: str) -> bool:
    """Check whether a file looks like a text file by extension."""
    name = os.path.basename(path).lower()
    stem, ext = os.path.splitext(name)
    return ext in TEXT_EXTENSIONS or stem in TEXT_EXTENSIONS or name in TEXT_EXTENSIONS


def _extract_keywords(text: str) -> set[str]:
    """Extract lowercase alphanumeric keywords from text, splitting compound identifiers."""
    raw = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text))
    lower_raw = {w.lower() for w in raw}
    subwords = set()
    for w in raw:
        subwords.update(p.lower() for p in w.split("_") if p)
        subwords.update(p.lower() for p in re.findall(r"[a-z]+|[A-Z][a-z]*", w) if p)
    return lower_raw | subwords


def _token_count(text: str) -> int:
    """Rough token count for context budgeting."""
    return len(text.split())


def _dot_product(a: list[float], b: list[float]) -> float:
    """Dot product of two equal-length vectors."""
    return sum(x * y for x, y in zip(a, b))


def get_embeddings(
    texts: list[str],
    model: str = DEFAULT_EMBED_MODEL,
) -> list[list[float]] | None:
    """
    Generate embeddings via the Ollama /api/embed endpoint.

    Args:
        texts: List of strings to embed.
        model: Embedding model name.

    Returns:
        List of embedding vectors, or None if the request fails.
    """
    if not texts:
        return []

    all_embeddings = []
    try:
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            payload = {"model": model, "input": batch}
            data = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            req = urllib.request.Request(
                OLLAMA_EMBED_URL, data=data, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req) as response:
                body = response.read().decode("utf-8")
                result = json.loads(body)
                all_embeddings.extend(result.get("embeddings", []))
        return all_embeddings
    except urllib.error.HTTPError as e:
        print(
            f"Warning: embedding API error {e.code} (model '{model}' may not be pulled). "
            "Falling back to keyword search.",
            file=sys.stderr,
        )
        return None
    except urllib.error.URLError as e:
        print(
            f"Warning: failed to connect to Ollama embedding endpoint: {e}. "
            "Falling back to keyword search.",
            file=sys.stderr,
        )
        return None
    except json.JSONDecodeError as e:
        print(
            f"Warning: failed to parse embedding response: {e}. "
            "Falling back to keyword search.",
            file=sys.stderr,
        )
        return None


def index_directory(
    root_dir: str,
    index_file: str = ".ollama_index.json",
    max_depth: int = 10,
    file_limit: int = 200,
    embed_model: str | None = None,
) -> dict:
    """
    Recursively scan a directory and build a JSON index of text files.

    Args:
        root_dir: Directory to scan.
        index_file: Path where the index JSON will be written.
        max_depth: Maximum directory depth to recurse.
        file_limit: Maximum number of files to index.
        embed_model: Optional Ollama embedding model to compute semantic vectors.

    Returns:
        The index dictionary.
    """
    entries = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Prune skip directories
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]
        current_depth = dirpath.count(os.sep) - root_dir.count(os.sep)
        if current_depth >= max_depth:
            dirnames[:] = []

        for filename in filenames:
            if count >= file_limit:
                break
            if filename in SKIP_FILES or filename.startswith(".") and not _is_text_file(filename):
                continue

            filepath = os.path.join(dirpath, filename)
            if not _is_text_file(filepath):
                continue

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except OSError:
                continue

            preview = content[:2000]
            keywords = list(_extract_keywords(preview))[:100]
            tokens = _token_count(content)

            rel_path = os.path.relpath(filepath, root_dir)
            entries.append(
                {
                    "path": filepath,
                    "rel_path": rel_path,
                    "tokens": tokens,
                    "keywords": keywords,
                    "preview": preview,
                    "embedding": None,
                }
            )
            count += 1

    if embed_model and entries:
        previews = [e["preview"] for e in entries]
        embeddings = get_embeddings(previews, model=embed_model)
        if embeddings:
            for entry, vec in zip(entries, embeddings):
                entry["embedding"] = vec
            print(
                f"Computed {len(embeddings)} semantic embeddings via '{embed_model}'.",
                file=sys.stderr,
            )
        else:
            print(
                "Embedding computation failed; index will use keyword search only.",
                file=sys.stderr,
            )

    index = {
        "root": os.path.abspath(root_dir),
        "file_count": len(entries),
        "entries": entries,
    }

    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Indexed {len(entries)} files into {index_file}", file=sys.stderr)
    return index


def load_index(index_file: str) -> dict | None:
    """Load an existing index JSON."""
    if not os.path.isfile(index_file):
        return None
    try:
        with open(index_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def search_index(
    index: dict,
    query: str,
    top_n: int = 5,
    max_context_tokens: int = 4000,
) -> list[dict]:
    """
    Score index entries against a query and return the most relevant ones
    using keyword overlap.

    Args:
        index: The index dictionary.
        query: User prompt to match against.
        top_n: Maximum number of files to return.
        max_context_tokens: Budget for total token count of returned previews.

    Returns:
        List of selected entry dicts with a 'score' key added.
    """
    query_keywords = _extract_keywords(query)
    scored = []
    for entry in index.get("entries", []):
        entry_keywords = set(entry.get("keywords", []))
        score = len(query_keywords & entry_keywords)
        if score > 0:
            scored.append({**entry, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    total_tokens = 0
    for entry in scored:
        entry_tokens = _token_count(entry["preview"])
        if total_tokens + entry_tokens > max_context_tokens:
            break
        selected.append(entry)
        total_tokens += entry_tokens
        if len(selected) >= top_n:
            break

    return selected


def semantic_search_index(
    index: dict,
    query: str,
    embed_model: str = DEFAULT_EMBED_MODEL,
    top_n: int = 5,
    max_context_tokens: int = 4000,
) -> list[dict]:
    """
    Score index entries against a query using vector (cosine) similarity.

    Args:
        index: The index dictionary.
        query: User prompt to match against.
        embed_model: Embedding model name for the query.
        top_n: Maximum number of files to return.
        max_context_tokens: Budget for total token count of returned previews.

    Returns:
        List of selected entry dicts with a 'score' key added, or empty list
        if embeddings are unavailable.
    """
    entries = index.get("entries", [])
    embedded_entries = [e for e in entries if e.get("embedding") is not None]
    if not embedded_entries:
        return []

    query_embedding = get_embeddings([query], model=embed_model)
    if not query_embedding:
        return []

    query_vec = query_embedding[0]
    scored = []
    for entry in embedded_entries:
        score = _dot_product(query_vec, entry["embedding"])
        scored.append({**entry, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    total_tokens = 0
    for entry in scored:
        entry_tokens = _token_count(entry["preview"])
        if total_tokens + entry_tokens > max_context_tokens:
            break
        selected.append(entry)
        total_tokens += entry_tokens
        if len(selected) >= top_n:
            break

    return selected


def read_files(file_paths: list[str]) -> str:
    """
    Read the contents of multiple files and format them as context.

    Args:
        file_paths: List of file paths to read.

    Returns:
        A formatted string with file contents, or an empty string if no files.
    """
    context_parts = []
    for path in file_paths:
        if not os.path.isfile(path):
            print(f"Warning: file not found, skipping: {path}", file=sys.stderr)
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            context_parts.append(
                f"--- FILE: {path} ---\n{content}\n--- END FILE: {path} ---"
            )
        except OSError as e:
            print(f"Warning: could not read {path}: {e}", file=sys.stderr)
    return "\n\n".join(context_parts)


def generate_code(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    stream: bool = False,
) -> str:
    """
    Send a prompt to the Ollama API and return the generated response.

    Args:
        prompt: The coding prompt to send.
        model: The Ollama model name to use.
        system: Optional system prompt override.
        stream: Whether to stream the response (default False for simplicity).

    Returns:
        The generated text from the model.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "system": system or PERSONAS["default"],
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    req = urllib.request.Request(OLLAMA_URL, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req) as response:
            body = response.read().decode("utf-8")
            result = json.loads(body)
            return result.get("response", "")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"Ollama API error {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Failed to connect to Ollama at {OLLAMA_URL}. "
            "Is Ollama running?"
        ) from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse Ollama response: {e}") from e


def main():
    parser = argparse.ArgumentParser(
        description="Generate code snippets using a local Ollama model."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="The coding prompt (e.g., 'write a Python function to reverse a string').",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "-p",
        "--persona",
        choices=list(PERSONAS.keys()),
        default="default",
        help="Predefined personality for the assistant.",
    )
    parser.add_argument(
        "-f",
        "--file",
        action="append",
        default=[],
        help="Path to a file to include as context (can be used multiple times).",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Optional custom system prompt to override the default/persona.",
    )
    parser.add_argument(
        "-d",
        "--vault-dir",
        default=None,
        help="Directory to index as a vault for contextual memory.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild of the vault index even if one exists.",
    )
    parser.add_argument(
        "--index-file",
        default=".ollama_index.json",
        help="Path to the index JSON file (default: .ollama_index.json).",
    )
    parser.add_argument(
        "--index-depth",
        type=int,
        default=10,
        help="Maximum recursion depth for vault indexing (default: 10).",
    )
    parser.add_argument(
        "--context-files",
        type=int,
        default=5,
        help="Maximum indexed files to include as context (default: 5).",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Use semantic (vector) search instead of keyword search for vault context.",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help=f"Ollama embedding model for semantic search (default: {DEFAULT_EMBED_MODEL}).",
    )

    args = parser.parse_args()

    prompt = args.prompt
    if not prompt:
        try:
            prompt = input("Enter your coding prompt: ").strip()
        except EOFError:
            prompt = ""

    if not prompt:
        print("Error: No prompt provided.", file=sys.stderr)
        sys.exit(1)

    index = None
    if args.vault_dir:
        if args.rebuild_index or not os.path.isfile(args.index_file):
            index = index_directory(
                args.vault_dir,
                index_file=args.index_file,
                max_depth=args.index_depth,
                embed_model=args.embed_model if args.semantic else None,
            )
        else:
            index = load_index(args.index_file)
            if index is None:
                index = index_directory(
                    args.vault_dir,
                    index_file=args.index_file,
                    max_depth=args.index_depth,
                    embed_model=args.embed_model if args.semantic else None,
                )
            elif args.semantic and not any(
                e.get("embedding") is not None for e in index.get("entries", [])
            ):
                print(
                    "Index lacks embeddings; rebuilding with semantic vectors...",
                    file=sys.stderr,
                )
                index = index_directory(
                    args.vault_dir,
                    index_file=args.index_file,
                    max_depth=args.index_depth,
                    embed_model=args.embed_model,
                )

    file_context_parts = []
    relevant = []

    if index:
        if args.semantic:
            relevant = semantic_search_index(
                index,
                prompt,
                embed_model=args.embed_model,
                top_n=args.context_files,
            )
            if not relevant:
                print(
                    "Semantic search returned no results; falling back to keyword search.",
                    file=sys.stderr,
                )
                relevant = search_index(index, prompt, top_n=args.context_files)
        else:
            relevant = search_index(index, prompt, top_n=args.context_files)

        for entry in relevant:
            file_context_parts.append(
                f"--- FILE: {entry['rel_path']} ---\n{entry['preview']}\n--- END FILE: {entry['rel_path']} ---"
            )

    explicit_context = read_files(args.file)
    if explicit_context:
        file_context_parts.append(explicit_context)

    if file_context_parts:
        prompt = (
            f"Here are some relevant files for context:\n\n"
            f"{'\n\n'.join(file_context_parts)}\n\n"
            f"Now, based on the above context, please: {prompt}"
        )

    system_prompt = args.system or PERSONAS.get(args.persona)

    print(f"Model: {args.model}")
    print(f"Persona: {args.persona}")
    if args.file:
        print(f"Explicit context files: {', '.join(args.file)}")
    if index and relevant:
        search_type = "semantic" if (args.semantic and relevant[0].get("embedding")) else "keyword"
        print(f"Indexed context files ({search_type}): {', '.join(e['rel_path'] for e in relevant)}")
    print("Generating code...\n")

    try:
        response = generate_code(prompt, model=args.model, system=system_prompt)
        print(response)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
