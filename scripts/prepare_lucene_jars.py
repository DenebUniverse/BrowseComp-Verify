#!/usr/bin/env python3
"""Download supplemental jars matching the installed Anserini's Lucene version."""

import argparse
import importlib.util
from pathlib import Path
import re
from urllib.request import urlopen
from zipfile import ZipFile


def installed_lucene_version():
    spec = importlib.util.find_spec("pyserini")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Install the retrieval/bm25 dependencies first")
    jar_dir = Path(next(iter(spec.submodule_search_locations))) / "resources/jars"
    jars = list(jar_dir.glob("anserini-*-fatjar.jar"))
    if len(jars) != 1:
        raise RuntimeError(f"Expected one installed Anserini fatjar in {jar_dir}")
    with ZipFile(jars[0]) as archive:
        name = "META-INF/maven/org.apache.lucene/lucene-core/pom.properties"
        if name not in archive.namelist():
            return None
        match = re.search(r"^version=(.+)$", archive.read(name).decode(), re.M)
        return match.group(1).strip() if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="explicit matching version if the fatjar lacks Maven metadata")
    parser.add_argument("--output_dir", default="external/lucene")
    args = parser.parse_args()
    detected = installed_lucene_version()
    if detected and args.version and detected != args.version:
        raise ValueError(f"Requested {args.version}, but installed Lucene is {detected}")
    version = args.version or detected
    if not version or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Could not identify Lucene version; supply --version matching your installed Anserini")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for artifact in ("lucene-highlighter", "lucene-queries", "lucene-memory"):
        target = output / f"{artifact}-{version}.jar"
        if target.exists():
            print(f"Already exists: {target}")
            continue
        url = f"https://repo.maven.apache.org/maven2/org/apache/lucene/{artifact}/{version}/{target.name}"
        with urlopen(url, timeout=60) as response:
            data = response.read()
        target.write_bytes(data)
        print(f"Downloaded {target}")


if __name__ == "__main__":
    main()
