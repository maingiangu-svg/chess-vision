"""Publish portable metrics without local workspace paths or machine hostnames."""
import argparse
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET


def portable(value, workspace):
    if isinstance(value, dict):
        return {key:portable(child,workspace) for key,child in value.items()}
    if isinstance(value, list):
        return [portable(child,workspace) for child in value]
    if not isinstance(value,str):
        return value
    for prefix in (str(workspace),str(workspace).replace("\\","/")):
        value = re.sub(re.escape(prefix)+r"[\\/]?", "", value, flags=re.IGNORECASE)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()
    workspace = Path.cwd().resolve()
    files = [path for path in args.reports.iterdir() if path.is_file()]
    if args.backup:
        args.backup.mkdir(parents=True,exist_ok=False)
        for path in files:
            shutil.copy2(path,args.backup/path.name)
    for path in files:
        if path.suffix == ".json":
            data = portable(json.loads(path.read_text(encoding="utf-8-sig")),workspace)
            path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        elif path.suffix == ".xml":
            tree = ET.parse(path)
            for element in tree.iter():
                element.attrib.pop("hostname",None)
                if element.text:
                    element.text = portable(element.text,workspace)
            tree.write(path,encoding="utf-8",xml_declaration=True)
        elif path.suffix in (".txt", ".md", ".jsonl"):
            path.write_text(portable(path.read_text(encoding="utf-8-sig"),workspace),encoding="utf-8")


if __name__ == "__main__":
    main()
