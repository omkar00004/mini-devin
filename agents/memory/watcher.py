# agents/memory/watcher.py

import os
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from semantic import index_file

# Only watch these file types — skip yaml, json, txt etc
WATCHED_EXTENSIONS = {".py", ".js", ".ts", ".go", ".java", ".rs"}


class CodebaseEventHandler(FileSystemEventHandler):
    """
    Handles file system events.
    watchdog calls on_modified/on_created whenever a file changes.
    We re-index that specific file immediately.
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        # Debounce: track last index time per file
        # Problem without this: editor saves a file 5 times in 100ms
        # (temp files, autosave, final save) → we'd index 5 times
        # Solution: ignore events within 1 second of last index
        self._last_indexed: dict[str, float] = {}
        self._lock = threading.Lock()

    def on_modified(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def _handle(self, abs_path: str):
        # Check extension
        if not any(abs_path.endswith(ext) for ext in WATCHED_EXTENSIONS):
            return

        # Debounce — skip if indexed within last 1 second
        now = time.time()
        with self._lock:
            last = self._last_indexed.get(abs_path, 0)
            if now - last < 1.0:
                return
            self._last_indexed[abs_path] = now

        # Re-index this specific file only
        rel_path = os.path.relpath(abs_path, self.base_dir)
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
            n = index_file(rel_path, content)
            print(f"  [watcher] {rel_path} → {n} chunks re-indexed")
        except Exception as e:
            print(f"  [watcher] failed to index {rel_path}: {e}")


def start_watcher(directory: str) -> Observer:
    """
    Start file watcher in background.
    Returns the Observer so caller can stop it cleanly.
    Non-blocking — returns immediately.
    """
    handler  = CodebaseEventHandler(directory)
    observer = Observer()
    observer.schedule(handler, directory, recursive=True)
    observer.start()
    print(f"  [watcher] watching {directory}")
    return observer