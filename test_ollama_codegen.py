#!/usr/bin/env python3
"""Unit tests for ollama_codegen indexing and context retrieval logic."""

import json
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ollama_codegen as og


class TestHelpers(unittest.TestCase):
    def test_is_text_file_by_extension(self):
        self.assertTrue(og._is_text_file("main.py"))
        self.assertTrue(og._is_text_file("script.js"))
        self.assertTrue(og._is_text_file("README.md"))
        self.assertFalse(og._is_text_file("image.png"))
        self.assertFalse(og._is_text_file("binary.exe"))

    def test_is_text_file_hidden(self):
        self.assertTrue(og._is_text_file(".gitignore"))
        self.assertTrue(og._is_text_file(".env.local"))

    def test_extract_keywords_basic(self):
        text = "def hello_world(): return 42"
        result = og._extract_keywords(text)
        self.assertIn("def", result)
        self.assertIn("hello_world", result)
        self.assertIn("return", result)
        self.assertNotIn("42", result)

    def test_extract_keywords_lowercase(self):
        text = "HelloWorld FooBar"
        result = og._extract_keywords(text)
        self.assertIn("helloworld", result)
        self.assertIn("foobar", result)

    def test_token_count(self):
        self.assertEqual(og._token_count("one two three"), 3)
        self.assertEqual(og._token_count(""), 0)


class TestIndexing(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="ollama_test_")
        self.index_path = os.path.join(self.tmpdir, "test_index.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_file(self, rel_path, content):
        full = os.path.join(self.tmpdir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)

    def test_index_directory_basic(self):
        self._write_file("a.py", "def foo(): pass")
        self._write_file("b.js", "const x = 1;")
        index = og.index_directory(
            self.tmpdir,
            index_file=self.index_path,
            max_depth=5,
            file_limit=10,
        )
        self.assertEqual(index["file_count"], 2)
        self.assertEqual(len(index["entries"]), 2)
        paths = {e["rel_path"] for e in index["entries"]}
        self.assertEqual(paths, {"a.py", "b.js"})

    def test_index_skips_dirs(self):
        self._write_file("node_modules/pkg/index.js", "module.exports = {}")
        self._write_file("src/main.py", "print('hello')")
        index = og.index_directory(
            self.tmpdir,
            index_file=self.index_path,
            max_depth=5,
            file_limit=10,
        )
        paths = {e["rel_path"] for e in index["entries"]}
        self.assertIn("src/main.py", paths)
        self.assertNotIn("node_modules/pkg/index.js", paths)

    def test_index_skips_hidden_dirs(self):
        self._write_file(".hidden/secret.py", "password = '123'")
        self._write_file("visible.py", "x = 1")
        index = og.index_directory(
            self.tmpdir,
            index_file=self.index_path,
            max_depth=5,
            file_limit=10,
        )
        paths = {e["rel_path"] for e in index["entries"]}
        self.assertIn("visible.py", paths)
        self.assertNotIn(".hidden/secret.py", paths)

    def test_index_respects_depth(self):
        self._write_file("level1/a.py", "a = 1")
        self._write_file("level1/level2/b.py", "b = 2")
        self._write_file("level1/level2/level3/c.py", "c = 3")
        index = og.index_directory(
            self.tmpdir,
            index_file=self.index_path,
            max_depth=2,
            file_limit=10,
        )
        paths = {e["rel_path"] for e in index["entries"]}
        self.assertIn("level1/a.py", paths)
        self.assertIn("level1/level2/b.py", paths)
        self.assertNotIn("level1/level2/level3/c.py", paths)

    def test_index_respects_file_limit(self):
        for i in range(5):
            self._write_file(f"f{i}.py", f"x = {i}")
        index = og.index_directory(
            self.tmpdir,
            index_file=self.index_path,
            max_depth=5,
            file_limit=3,
        )
        self.assertEqual(index["file_count"], 3)

    def test_index_creates_json_file(self):
        self._write_file("a.py", "def foo(): pass")
        og.index_directory(
            self.tmpdir,
            index_file=self.index_path,
            max_depth=5,
            file_limit=10,
        )
        self.assertTrue(os.path.isfile(self.index_path))
        with open(self.index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["file_count"], 1)

    def test_index_keywords_populated(self):
        self._write_file("math.py", "def add(a, b): return a + b")
        index = og.index_directory(
            self.tmpdir,
            index_file=self.index_path,
            max_depth=5,
            file_limit=10,
        )
        entry = index["entries"][0]
        keywords = set(entry["keywords"])
        self.assertIn("def", keywords)
        self.assertIn("add", keywords)
        self.assertIn("return", keywords)


class _FakeResponse:
    """Minimal urllib response stand-in for testing."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestEmbeddings(unittest.TestCase):
    def test_dot_product(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        self.assertEqual(og._dot_product(a, b), 0.0)
        c = [1.0, 2.0, 3.0]
        d = [4.0, 5.0, 6.0]
        self.assertEqual(og._dot_product(c, d), 32.0)

    def test_get_embeddings_empty(self):
        result = og.get_embeddings([])
        self.assertEqual(result, [])

    def test_get_embeddings_success(self):
        payload = json.dumps(
            {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
        ).encode("utf-8")
        fake_ctx = unittest.mock.patch(
            "urllib.request.urlopen",
            return_value=_FakeResponse(payload),
        )
        fake_req = unittest.mock.patch("urllib.request.Request")
        with fake_ctx, fake_req:
            result = og.get_embeddings(["hello", "world"])
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 3)
        self.assertEqual(len(result[1]), 3)

    def test_get_embeddings_failure(self):
        import urllib.error
        fake_ctx = unittest.mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("No connection"),
        )
        fake_req = unittest.mock.patch("urllib.request.Request")
        with fake_ctx, fake_req:
            result = og.get_embeddings(["hello"])
        self.assertIsNone(result)


class TestSearchAndContext(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="ollama_test_")
        self.index_path = os.path.join(self.tmpdir, "test_index.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_file(self, rel_path, content):
        full = os.path.join(self.tmpdir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)

    def test_load_index(self):
        self._write_file("a.py", "x = 1")
        og.index_directory(self.tmpdir, index_file=self.index_path)
        loaded = og.load_index(self.index_path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["file_count"], 1)

    def test_load_index_missing(self):
        result = og.load_index(os.path.join(self.tmpdir, "no_such_file.json"))
        self.assertIsNone(result)

    def test_search_index_basic(self):
        self._write_file("auth.py", "def authenticate_user(): pass")
        self._write_file("utils.py", "def helper(): pass")
        index = og.index_directory(self.tmpdir, index_file=self.index_path)
        results = og.search_index(index, "authenticate user login", top_n=5)
        rel_paths = [r["rel_path"] for r in results]
        self.assertIn("auth.py", rel_paths)

    def test_search_index_ranking(self):
        self._write_file("auth.py", "def authenticate_user(): pass")
        self._write_file("db.py", "def query_database(): pass")
        self._write_file("ui.py", "def render_button(): pass")
        index = og.index_directory(self.tmpdir, index_file=self.index_path)
        results = og.search_index(index, "authenticate user login", top_n=5)
        self.assertEqual(results[0]["rel_path"], "auth.py")

    def test_search_index_respects_top_n(self):
        for i in range(10):
            self._write_file(f"f{i}.py", f"x{i} = {i}")
        index = og.index_directory(self.tmpdir, index_file=self.index_path)
        results = og.search_index(index, "x0 x1 x2", top_n=2)
        self.assertLessEqual(len(results), 2)

    def test_search_index_respects_token_budget(self):
        self._write_file("big.py", "x = 1\n" * 2000)
        self._write_file("small.py", "y = 2")
        index = og.index_directory(self.tmpdir, index_file=self.index_path)
        results = og.search_index(
            index, "x y", top_n=5, max_context_tokens=50
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["rel_path"], "small.py")

    def test_read_files_single(self):
        path = os.path.join(self.tmpdir, "test.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("hello world")
        result = og.read_files([path])
        self.assertIn("--- FILE:", result)
        self.assertIn("hello world", result)

    def test_read_files_missing(self):
        result = og.read_files([os.path.join(self.tmpdir, "missing.txt")])
        self.assertEqual(result, "")

    def test_read_files_multiple(self):
        p1 = os.path.join(self.tmpdir, "a.txt")
        p2 = os.path.join(self.tmpdir, "b.txt")
        with open(p1, "w", encoding="utf-8") as f:
            f.write("alpha")
        with open(p2, "w", encoding="utf-8") as f:
            f.write("beta")
        result = og.read_files([p1, p2])
        self.assertIn("alpha", result)
        self.assertIn("beta", result)

    def test_index_directory_with_embeddings(self):
        self._write_file("a.py", "def foo(): pass")
        payload = json.dumps({"embeddings": [[0.1, 0.2]]}).encode("utf-8")
        fake_ctx = unittest.mock.patch(
            "urllib.request.urlopen",
            return_value=_FakeResponse(payload),
        )
        fake_req = unittest.mock.patch("urllib.request.Request")
        with fake_ctx, fake_req:
            index = og.index_directory(
                self.tmpdir,
                index_file=self.index_path,
                embed_model="nomic-embed-text",
            )
        self.assertEqual(len(index["entries"]), 1)
        self.assertIsNotNone(index["entries"][0]["embedding"])

    def test_semantic_search_basic(self):
        index = {
            "entries": [
                {
                    "rel_path": "auth.py",
                    "preview": "def authenticate(): pass",
                    "tokens": 4,
                    "embedding": [1.0, 0.0],
                },
                {
                    "rel_path": "utils.py",
                    "preview": "def helper(): pass",
                    "tokens": 3,
                    "embedding": [0.0, 1.0],
                },
            ]
        }
        payload = json.dumps({"embeddings": [[0.9, 0.1]]}).encode("utf-8")
        fake_ctx = unittest.mock.patch(
            "urllib.request.urlopen",
            return_value=_FakeResponse(payload),
        )
        fake_req = unittest.mock.patch("urllib.request.Request")
        with fake_ctx, fake_req:
            results = og.semantic_search_index(index, "authenticate user", top_n=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["rel_path"], "auth.py")

    def test_semantic_search_no_embeddings(self):
        index = {
            "entries": [
                {
                    "rel_path": "a.py",
                    "preview": "x = 1",
                    "tokens": 3,
                    "embedding": None,
                }
            ]
        }
        results = og.semantic_search_index(index, "query", top_n=5)
        self.assertEqual(results, [])

    def test_semantic_search_fallback(self):
        self._write_file("auth.py", "def authenticate(): pass")
        self._write_file("utils.py", "def helper(): pass")
        index = og.index_directory(self.tmpdir, index_file=self.index_path)
        results = og.semantic_search_index(index, "authenticate", top_n=5)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
