import pandas as pd
import ast
from datetime import datetime
import io
import json
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
from urllib.parse import unquote
import openpyxl
import hashlib
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import Language,RecursiveCharacterTextSplitter
import httpx
import os
import concurrent.futures
import threading
import tiktoken as tt
import re
import ast
from typing import List, Tuple, Dict, Any, Optional, Union



# for local run only use below credential



_DEF_RE = re.compile(r'^(?P<indent>[ \t]*)def\s+(?P<name>[A-Za-z_]\w*)\s*\(', re.M)
_ASYNC_DEF_RE = re.compile(r'^(?P<indent>[ \t]*)async\s+def\s+(?P<name>[A-Za-z_]\w*)\s*\(', re.M)
_CLASS_RE = re.compile(r'^(?P<indent>[ \t]*)class\s+(?P<name>[A-Za-z_]\w*)\s*\(', re.M)
_DECORATOR_RE = re.compile(r'^(?P<indent>[ \t]*)@', re.M)
_IMPORT_RE = re.compile(r'^(?:from\s+[.\w]+\s+import\s+[^\n]+|import\s+[^\n]+)', re.M)


# Generating doc for each splitted code
class HttpClient:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            # time being path is hardcoded
            cacert_path = os.path.abspath('cacert.pem')  # enable this before commig to dev
            print("Path: ", cacert_path)

            
            
            # Check if the cacert.pem file exists
            if os.path.exists(cacert_path):
                cls._client = httpx.Client(verify=cacert_path)
            else:
                print(f"Certificate file not found: {cacert_path}. Proceeding without custom certificate.")
                cls._client = httpx.Client()  # Default behavior without specifying 'verify'
        
        return cls._client


client = HttpClient.get_client()


# for hashcode generation
def generate_hash(text):
    hash_object = hashlib.sha256(text.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex



# Function to initialize BlobServiceClient
def get_blob_service_client(tenant_id, client_id, client_secret, storage_account_name):
    account_url = f"https://{storage_account_name}.blob.core.windows.net"
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
    return blob_service_client


# This is for when we commiting changes 
llm = AzureChatOpenAI(
    openai_api_key = os.environ["OPENAI_API_KEY"],
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version = os.environ["OPENAI_API_VERSION"],
    azure_deployment = os.environ["AZURE_OPENAI_SUMMARIZATION_MODEL"],
    temperature = 0,
    http_client=client
)



def get_completion_azure_ml(prompt=None):
    """Function to generate response from gpt based on user query, context and model parameters"""
    try:
        if prompt!= None:
            messages=[
            {"role": "system", "content": "You are expert Python developer"},
            {"role": "user", "content":"Deeply understand the given python code and extract the meaning of it in concise and precise way, the given python script is : "+prompt}
            ]
            ai_message = llm.invoke(messages)
        else:
            messages=[
            {"role": "system", "content": "You are expert Python developer"},
            {"role": "user", "content": "Check if prompt content is empty then say 'No Description' for empty prompt, the given prompt is:"+prompt}
            ]
            ai_message = llm.invoke(messages)
        return (ai_message.content)
    except Exception as e:
        print(f"Unable to extract the meaning of the code forthe chunk in {prompt}; Error: {str(e)}")
        return None
    
def process_class_definition(node, lines, file_content, imports_text):
    """
    Processes a class definition node, extracting its header, methods, and context.

    Args:
        node (ast.ClassDef): The class definition node to process.
        lines (list): List of lines from the source file.
        file_content (str): Full content of the source file.
        imports_text (str): Text of the imports in the file.

    Returns:
        chunk (str): The formatted chunk of code for the class.
        chunk_context (dict): Context information for the class.
    """
    # Find all method nodes
    method_nodes = [item for item in node.body if isinstance(item, ast.FunctionDef)]
    if not method_nodes:
        return

    # Find first method to determine where class header ends
    first_method = min(method_nodes, key=lambda n: n.lineno)
    class_header_start = node.lineno - 1
    class_header_end = first_method.lineno - 1
    class_header = lines[class_header_start:class_header_end]
    class_header_text = "\n".join(class_header)

    # Find __init__ (if present)
    init_node = next((item for item in method_nodes if item.name == "__init__"), None)
    init_text = ast.get_source_segment(file_content, init_node) if init_node else ""

    for method_node in method_nodes:
        if method_node.name == "__init__":
            continue
        method_text = ast.get_source_segment(file_content, method_node)
        method_signature = ""
        if method_text:
            for line in method_text.splitlines():
                if line.strip().startswith("def "):
                    method_signature = line.strip()
                    break
        chunk = "\n".join([
            imports_text,
            "",
            class_header_text,
            init_text,
            method_text
        ]).strip()
        chunk_context = {
            "imports_text": imports_text,
            "class_header_text": class_header_text,
            "method_signature": method_signature
        }

        return chunk, chunk_context
    
def process_function_definition(node, file_content, imports_text):
    """
    Processes a function definition node, extracting its text, signature, and context.

    Args:
        node (ast.FunctionDef): The function definition node to process.
        file_content (str): Full content of the source file.
        imports_text (str): Text of the imports in the file.
        chunks (list): List to store processed chunks.
        chunk_contexts (list): List to store context information for each chunk.

    Returns:
        chunk (str): The formatted chunk of code for the function.
        chunk_context (dict): Context information for the function.
    """
    func_text = ast.get_source_segment(file_content, node)
    func_signature = ""
    if func_text:
        for line in func_text.splitlines():
            if line.strip().startswith("def "):
                func_signature = line.strip()
                break
    chunk = "\n".join([
        imports_text,
        "",
        func_text
    ]).strip()
    
    chunk_context = {
        "imports_text": imports_text,
        "class_header_text": "",
        "method_signature": func_signature
    }
    return chunk, chunk_context


def _first_line(text: str, limit: int = 160) -> str:
    line = re.sub(r'\s+', ' ', (text or '').strip())
    return (line[:limit] + '…') if len(line) > limit else line


def _collect_imports_text(src: str) -> str:
    imports = _IMPORT_RE.findall(src)
    return "\n".join(imports)


def _normalize_eol_quotes(src: str) -> str:
    return (src.replace('\r\n', '\n').replace('\r', '\n')
               .replace("“", '"').replace("”", '"')
               .replace("‘", "'").replace("’", "'"))


def _slice_by_lines(src_lines: List[str], start: int, end: int) -> str:
    # start/end are 1-based inclusive/exclusive like AST end_lineno+1
    start = max(1, start)
    end = min(len(src_lines) + 1, end if end else len(src_lines) + 1)
    return "\n".join(src_lines[start-1:end-1])


def _include_decorators_start(node: Union[ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
    min_line = node.lineno
    for dec in getattr(node, "decorator_list", []) or []:
        if hasattr(dec, "lineno"):
            min_line = min(min_line, dec.lineno)
    return min_line


def _docstring_of(node: Union[ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
    try:
        doc = ast.get_docstring(node, clean=True)
        return _first_line(doc) if doc else None
    except Exception:
        return None


def _summarize(kind: str, qualname: str, code: str, node: Optional[ast.AST], default_hint: str = "") -> str:
    if node is not None:
        doc = _docstring_of(node)
        if doc:
            return f"{kind} {qualname}: {doc}"
    if default_hint:
        return f"{kind} {qualname} — {default_hint}"
    return f"{kind} {qualname}"


def _decorators_list(node: Union[ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef], src: str, src_lines: List[str]) -> List[str]:
    decs = []
    for d in getattr(node, "decorator_list", []) or []:
        try:
            # reconstruct decorator text from lines (best-effort)
            start = getattr(d, "lineno", None)
            end = getattr(d, "end_lineno", None)
            if start is not None and end is not None:
                decs.append(_slice_by_lines(src_lines, start, end).strip())
        except Exception:
            pass
    return decs

def _fix_unterminated_string_line(line: str) -> str:
    single_open = 0
    double_open = 0
    i = 0
    while i < len(line):
        ch = line[i]
        if ch in ("'", '"') and not (i > 0 and line[i-1] == "\\"):
            if ch == "'":
                single_open ^= 1
            else:
                double_open ^= 1
        i += 1
    if single_open and not double_open:
        return line + "'"
    if double_open and not single_open:
        return line + '"'
    if single_open or double_open:
        return line + '"'
    return line


def _fix_orphan_try_blocks(lines: List[str]) -> List[str]:
    result = lines[:]
    i = 0
    while i < len(result):
        line = result[i]
        m = re.match(r'^([ \t]*)try:\s*(#.*)?$', line)
        if not m:
            i += 1
            continue
        ind = m.group(1); ind_len = len(ind.replace("\t","    "))
        j = i + 1
        saw_handler = False
        while j < len(result):
            l2 = result[j]
            if not l2.strip() or re.match(r'^[ \t]*#', l2):  # blank/comment
                j += 1; continue
            m2 = re.match(r'^([ \t]*)(except\b|finally\b)', l2)
            if m2 and len(m2.group(1).replace("\t","    ")) == ind_len:
                saw_handler = True; break
            curr = len(re.match(r'^([ \t]*)', l2).group(1).replace("\t","    "))
            if curr <= ind_len:
                break
            j += 1
        if not saw_handler:
            insertion = f"{ind}except Exception:\n{ind}    pass"
            result.insert(j, insertion)
            i = j + 1
        else:
            i = j + 1
    return result


def _sanitize_once(src: str, err: Optional[SyntaxError]) -> str:
    if not err:
        return src
    msg = (err.msg or "").lower()
    lineno = err.lineno or 0
    lines = src.splitlines()
    if "unterminated string literal" in msg or "eol while scanning string literal" in msg:
        if 1 <= lineno <= len(lines):
            lines[lineno-1] = _fix_unterminated_string_line(lines[lineno-1])
        return "\n".join(lines)
    if "expected 'except' or 'finally' block" in msg:
        return "\n".join(_fix_orphan_try_blocks(lines))
    # generic rescue for try:-like messages
    window = "\n".join(lines[max(0, lineno-3):min(len(lines), lineno+2)])
    if "expected" in msg and "block" in msg and ("try" in window):
        return "\n".join(_fix_orphan_try_blocks(lines))
    return src


def _sanitize_source(src: str, max_passes: int = 3) -> str:
    current = src
    last_err: Optional[SyntaxError] = None
    for _ in range(max_passes):
        try:
            ast.parse(current)
            return current
        except SyntaxError as e:
            last_err = e
            current = _sanitize_once(current, e)
        except Exception:
            break
    try:
        ast.parse(current)
        return current
    except Exception:
        # last-ditch: try fix orphan tries
        return "\n".join(_fix_orphan_try_blocks(current.splitlines()))


class _Collector(ast.NodeVisitor):
    """
    Collects classes, functions, methods, async defs, nested items (optional),
    and optionally top-level executable code (main guard) ranges.
    """
    def __init__(self, src: str, include_inner: bool, include_toplevel_code: bool):
        self.src = src
        self.lines = src.splitlines()
        self.include_inner = include_inner
        self.include_toplevel_code = include_toplevel_code
        self.items: List[Dict[str, Any]] = []
        self.parents: List[str] = []  # class/def stack for qualname

    def _add_item(self, kind: str, name: str, node: Union[ast.AST, ast.stmt], parent: Optional[str]):
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return
        start = node.lineno
        end = node.end_lineno + 1  # exclusive
        # ensure decorators are included
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min(start, _include_decorators_start(node))

        code = _slice_by_lines(self.lines, start, end)
        qualname = ".".join([*(self.parents or ([] if parent is None else [parent])), name]).strip(".")
        decs = _decorators_list(node, self.src, self.lines) if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) else []

        ctx = {
            "type": kind,
            "name": name,
            "qualname": qualname or name,
            "parent": ".".join(self.parents) if self.parents else None,
            "decorators": decs,
            "lineno_start": start,
            "lineno_end_exclusive": end,
            "summary": _summarize(kind, qualname or name, code, node,
                                  default_hint=("initializer" if name == "__init__" else "")),
        }
        self.items.append({"code": code, "context": ctx})

    # --- module-level docstring and top-level code guard ---
    def visit_Module(self, node: ast.Module):
        # Module docstring chunk
        doc = _docstring_of(node)
        if doc:
            # capture exact module docstring code
            if node.body and isinstance(node.body[0], (ast.Expr,)):
                first = node.body[0]
                if hasattr(first, "lineno") and hasattr(first, "end_lineno"):
                    code = _slice_by_lines(self.lines, first.lineno, first.end_lineno + 1)
                    self.items.append({
                        "code": code,
                        "context": {
                            "type": "module_docstring",
                            "name": "<module-doc>",
                            "qualname": "<module-doc>",
                            "parent": None,
                            "decorators": [],
                            "lineno_start": first.lineno,
                            "lineno_end_exclusive": first.end_lineno + 1,
                            "summary": f"module doc: {doc}",
                        }
                    })
        # Optional: capture the `if __name__ == "__main__"` block as top-level code
        if self.include_toplevel_code:
            for n in node.body:
                if isinstance(n, ast.If):
                    try:
                        # detect __name__ == "__main__"
                        test_src = ast.get_source_segment(self.src, n.test) or ""
                        if "__name__" in test_src and "__main__" in test_src:
                            self._add_item("top_level_code", "main_guard", n, parent=None)
                    except Exception:
                        pass

        # Continue traversal
        self.generic_visit(node)

    # --- classes and functions ---
    def visit_ClassDef(self, node: ast.ClassDef):
        self._add_item("class", node.name, node, parent=None if not self.parents else self.parents[-1])
        self.parents.append(node.name)
        # Visit body to collect methods and nested classes/funcs
        for sub in node.body:
            if isinstance(sub, ast.FunctionDef):
                self._add_item("method", sub.name, sub, parent=node.name)
                if self.include_inner:
                    # Collect inner defs in method
                    self._visit_inner(sub, parent_qual=node.name + "." + sub.name)
            elif isinstance(sub, ast.AsyncFunctionDef):
                self._add_item("method", sub.name, sub, parent=node.name)
                if self.include_inner:
                    self._visit_inner(sub, parent_qual=node.name + "." + sub.name)
            elif isinstance(sub, ast.ClassDef):
                # nested class
                self.visit_ClassDef(sub)
        self.parents.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Only top-level if no parents; inner functions handled only if include_inner
        if not self.parents:
            self._add_item("function", node.name, node, parent=None)
        if self.include_inner:
            self._visit_inner(node, parent_qual=node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if not self.parents:
            self._add_item("function", node.name, node, parent=None)
        if self.include_inner:
            self._visit_inner(node, parent_qual=node.name)

    # recurse inner defs
    def _visit_inner(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], parent_qual: str):
        for sub in ast.iter_child_nodes(node):
            if isinstance(sub, ast.FunctionDef):
                self._add_item("inner_function", sub.name, sub, parent=parent_qual)
                self._visit_inner(sub, parent_qual=parent_qual + "." + sub.name)
            elif isinstance(sub, ast.AsyncFunctionDef):
                self._add_item("inner_function", sub.name, sub, parent=parent_qual)
                self._visit_inner(sub, parent_qual=parent_qual + "." + sub.name)
            elif isinstance(sub, ast.ClassDef):
                # inner class inside function
                self._add_item("inner_class", sub.name, sub, parent=parent_qual)
                # collect its methods too
                saved = self.parents[:]
                self.parents = parent_qual.split(".")
                self.visit_ClassDef(sub)
                self.parents = saved


def _include_leading_decorators(src: str, start_pos: int) -> int:
    line_start = src.rfind('\n', 0, start_pos) + 1
    pos = line_start
    while True:
        prev_nl = src.rfind('\n', 0, pos - 1)
        candidate_line_start = 0 if prev_nl == -1 else prev_nl + 1
        line = src[candidate_line_start:pos]
        if _DECORATOR_RE.match(line):
            pos = candidate_line_start
            if pos == 0:
                return 0
            continue
        break
    return pos


def _heuristic_split(src: str) -> List[Dict[str, Any]]:
    """
    Coarse splitter when AST fails: collects top-level classes and functions (sync/async),
    includes decorators, and then tries to split class bodies for methods.
    """
    items = []
    matches = []
    for m in _CLASS_RE.finditer(src):
        matches.append((m.start(), "class", m.group("name")))
    for m in _DEF_RE.finditer(src):
        matches.append((m.start(), "function", m.group("name")))
    for m in _ASYNC_DEF_RE.finditer(src):
        matches.append((m.start(), "function", m.group("name")))
    matches.sort(key=lambda x: x[0])

    for i, (pos, kind, name) in enumerate(matches):
        start = _include_leading_decorators(src, pos)
        end = matches[i+1][0] if i+1 < len(matches) else len(src)
        code = src[start:end].rstrip()
        items.append({
            "code": code,
            "context": {
                "type": kind,
                "name": name,
                "qualname": name,
                "parent": None,
                "decorators": [],
                "lineno_start": None,
                "lineno_end_exclusive": None,
                "summary": f"{kind} {name} (heuristic)",
            }
        })
    return items


def chunk_python_file(
    file_content: str,
    *,
    sanitize: bool = True,
    include_inner_functions: bool = True,
    include_top_level_code: bool = True
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Universal Python chunker:
      - AST-first extraction of classes (incl. nested), methods (sync/async), functions (top-level and inner*),
        decorators, docstrings, and the main guard block (optional).
      - Sanitizer to auto-fix common syntax issues (orphan try, unterminated strings, smart quotes).
      - Regex fallback when AST still fails.

    Returns:
      chunks: List[str]              # code snippets (each isolated chunk)
      contexts: List[Dict[str, Any]] # metadata per chunk:
        {
          'type': 'class'|'method'|'function'|'inner_function'|'inner_class'|'module_docstring'|'top_level_code',
          'name': str,
          'qualname': str,
          'parent': Optional[str],
          'decorators': List[str],
          'lineno_start': int | None,
          'lineno_end_exclusive': int | None,
          'summary': str,
        }
    """
    # Normalize & optionally sanitize
    source = _normalize_eol_quotes(file_content)
    source_for_parse = _sanitize_source(source) if sanitize else source

    # Try AST path
    try:
        tree = ast.parse(source_for_parse)
        collector = _Collector(source_for_parse, include_inner=include_inner_functions, include_toplevel_code=include_top_level_code)
        collector.visit(tree)

        # Always prefix each chunk with collected imports to keep context, like your original
        imports_text = _collect_imports_text(source_for_parse)
        chunks, ctxs = [], []
        for it in collector.items:
            code = it["code"]
            chunk = f"{imports_text}\n\n{code}".strip() if imports_text else code
            chunks.append(chunk)
            ctxs.append(it["context"])
        return chunks, ctxs

    except SyntaxError as e:
        err_msg = f"AST parsing failed: {e.msg} at line {e.lineno}, col {e.offset}"
    except Exception as e:
        err_msg = f"AST parsing failed: {e}"

    # Fallback (heuristic)
    imports_text = _collect_imports_text(source_for_parse)
    coarse = _heuristic_split(source_for_parse)
    chunks_f, ctxs_f = [], []

    # Add a synthetic file entry to carry the error info
    chunks_f.append(source_for_parse)
    ctxs_f.append({
        "type": "file",
        "name": "<module>",
        "qualname": "<module>",
        "parent": None,
        "decorators": [],
        "lineno_start": None,
        "lineno_end_exclusive": None,
        "summary": _first_line(err_msg) if err_msg else "Heuristic parse of module",
    })

    for it in coarse:
        code = it["code"]
        chunk = f"{imports_text}\n\n{code}".strip() if imports_text else code
        chunks_f.append(chunk)
        ctxs_f.append(it["context"])

    return chunks_f, ctxs_f

# Handling token limit here 
def handle_token_limit(text, filename, use_case, parent_path, imports_text="", class_header_text="", method_signature="", init_text="", token_limit=6500, _is_first=True):
    """
    Recursively splits a chunk into as many parts as needed, each with context, so that no part exceeds the token limit.
    The context (imports, class header, method signature, __init__ method) is included in every split part.
    """
    # Remove any existing filename prefix if present
    text = text.replace(f"filename: {filename}", "").strip()

    # Include the __init__ method in the context if provided
    context = "\n".join(filter(None, [imports_text, class_header_text, method_signature, init_text, ""]))
    tokens = tt.get_encoding("cl100k_base").encode(text)

    # Update handle_token_limit to handle function splitting with context for each chunk
    if _is_first or not _is_first:
        # Always include the context (imports, class header, method signature, and __init__ method) in every chunk
        text_with_context = f"{context}{text}".strip()

        # Check for duplicate context and remove it if found
        if text_with_context.count(context) > 1:
            text_with_context = text_with_context.replace(context, "", 1).strip()

        text = text_with_context

    # Adjust token count to include context size
    context_tokens = tt.get_encoding("cl100k_base").encode(context)
    total_token_count = len(tokens) + len(context_tokens)

    if total_token_count <= token_limit:
        return [
            {
                "chunk": f"filename: {filename}\n{text}",
                "chunk_hash": generate_hash(text),
                "use_case": use_case,
                "row": -1,
                "parent_path": parent_path,
                "sheet_name": "",
                "split_part": 1,
            }
        ]
    else:
        # Split based on tokens instead of characters
        mid = len(tokens) // 2
        part1_tokens = tokens[:mid]
        part2_tokens = tokens[mid:]

        # Decode tokens back to text
        part1 = tt.get_encoding("cl100k_base").decode(part1_tokens)
        part2 = tt.get_encoding("cl100k_base").decode(part2_tokens)

        # Add context to each split chunk
        part1_with_context = f"{context}{part1}".strip()
        part2_with_context = f"{context}{part2}".strip()

        results = []
        results.extend(
            handle_token_limit(
                part1_with_context,
                filename,
                use_case,
                parent_path,
                imports_text,
                class_header_text,
                method_signature,
                init_text,
                token_limit,
                False,
            )
        )
        results.extend(
            handle_token_limit(
                part2_with_context,
                filename,
                use_case,
                parent_path,
                imports_text,
                class_header_text,
                method_signature,
                init_text,
                token_limit,
                False,
            )
        )

        for i, chunk in enumerate(results):
            chunk["split_part"] = i + 1

        print(f"Number of chunks after splitting: {len(results)}")
        return results



def script_splitter(blob_uri):
    # Environment variables for authentication
    tenant_id = os.environ["TENET_ID"]
    client_id = os.environ["CLIENT_ID"]
    client_secret = os.environ["CLIENT_SECRET"]
    storage_account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = "knowledge-mining"

    # Authenticate and get BlobServiceClient
    blob_service_client = get_blob_service_client(tenant_id, client_id, client_secret, storage_account_name)

    # Parse blob URI
    unquoted_blob_uri = unquote(blob_uri)
    parent_path_with_container_name = unquoted_blob_uri.split(".blob.core.windows.net/")[1]
    use_case = parent_path_with_container_name.split("/")[1]

    # Get blob content
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=unquoted_blob_uri.split(container_name)[1]
    )

    if unquoted_blob_uri.endswith('.py'):
        blob_data = blob_client.download_blob().content_as_text()
        chunks,chunk_contexts = chunk_python_file(blob_data)

        new_chunks = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            # Map futures to original chunks
            future_to_chunk = {
                executor.submit(get_completion_azure_ml, chunk): (chunk, context)
                for chunk, context in zip(chunks, chunk_contexts)
            }

            for future in concurrent.futures.as_completed(future_to_chunk):
                original_chunk, context = future_to_chunk[future]
                result = str(future.result())

                 # Create the chunk content without filename prefix
                temp_chunk = (
                    original_chunk
                    + "\n# The python code description is:\n"
                    + result
                )

                # Ensure proper context handling to avoid duplication
                # Extract context from current chunk
                imports_text = context.get("imports_text", "")
                class_header_text = context.get("class_header_text", "")
                method_signature = context.get("method_signature", "")
                context_str = "\n".join(filter(None, [imports_text, class_header_text, method_signature, ""]))

                # Remove context from temp_chunk before passing to handle_token_limit
                if temp_chunk.startswith(context_str):
                    code_only = temp_chunk[len(context_str):].lstrip("\n")
                else:
                    code_only = temp_chunk  # Fallback if context_str is not found


                # Pass the cleaned chunk to handle_token_limit
                tokens = tt.get_encoding("cl100k_base").encode(code_only)
                if len(tokens) > 6500:
                    
                    split_chunks = handle_token_limit(
                        code_only,  # Pass the cleaned chunk
                        str(blob_uri.split('/')[-1]), 
                        use_case, 
                        parent_path_with_container_name,
                        imports_text=imports_text,
                        class_header_text=class_header_text,
                        method_signature=method_signature,
                        token_limit=6500  # Explicitly set the token limit to 7000
                    )
                    new_chunks.extend(split_chunks)
                else:
                    # Do not add context here; handle_token_limit will handle it
                    data = {
                        "chunk": f"filename: {blob_uri.split('/')[-1]}\n{code_only}",
                        "chunk_hash": generate_hash(code_only),
                        "use_case": use_case,
                        "row": -1,
                        "parent_path": parent_path_with_container_name,
                        "sheet_name": "",
                    }
                    new_chunks.append(data)

        return new_chunks
