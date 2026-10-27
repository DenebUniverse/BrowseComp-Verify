"""Small shared helpers; retrieval dependencies are loaded only when needed."""

import glob
import os
from pathlib import Path

from .data import load_dataset

DEVELOPER_CONTENT = """You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.
Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()


def load_bcplus_data(data_path, include_pid=False, task="solve"):
    if include_pid:
        raise ValueError("Document relevance judgments are not part of the DIAMOND schema")
    return load_dataset(data_path, task=task)


def setup_jar(extra_dir=None):
    """Initialize the JVM classpath before pyserini.search or jnius is imported."""
    import jnius_config
    import pyserini.setup as setup

    jar_dir = Path(setup.__file__).parent / "resources" / "jars"
    candidates = sorted(jar_dir.glob("anserini-*-fatjar.jar"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one Anserini fatjar in {jar_dir}; found {len(candidates)}")
    extras = sorted(glob.glob(os.path.join(extra_dir, "*.jar"))) if extra_dir else []
    jnius_config.add_options("-Djava.awt.headless=true")
    jnius_config.set_classpath(str(candidates[0]), *extras)
