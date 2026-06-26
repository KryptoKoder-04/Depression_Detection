import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


PPTX_PATH = Path(r"C:\Users\jiten\Downloads\TSFFM_Depression_Detection (1).pptx")
OUT_DIR = Path(r"C:\Users\jiten\btp_final2\outputs\midsem_extract")
OUT_PATH = OUT_DIR / "midsem_ppt_text.txt"


def natural_slide_key(name: str):
    match = re.search(r"slide(\d+)\.xml$", name)
    return int(match.group(1)) if match else 10**9


def extract_slide_text(xml_bytes: bytes):
    root = ET.fromstring(xml_bytes)
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    lines = []
    for paragraph in root.findall(".//a:p", ns):
        runs = []
        for node in paragraph.findall(".//a:t", ns):
            if node.text:
                runs.append(node.text)
        text = "".join(runs).strip()
        if text:
            lines.append(text)
    return lines


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slide_entries = []
    with zipfile.ZipFile(PPTX_PATH) as zf:
        slide_names = sorted(
            [n for n in zf.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)],
            key=natural_slide_key,
        )
        media_names = sorted([n for n in zf.namelist() if n.startswith("ppt/media/")])
        for idx, name in enumerate(slide_names, start=1):
            lines = extract_slide_text(zf.read(name))
            slide_entries.append((idx, lines))

    with OUT_PATH.open("w", encoding="utf-8") as f:
        f.write(f"Source PPTX: {PPTX_PATH}\n")
        f.write(f"Slides extracted: {len(slide_entries)}\n\n")
        for idx, lines in slide_entries:
            f.write(f"--- Slide {idx} ---\n")
            if lines:
                for line in lines:
                    f.write(f"{line}\n")
            else:
                f.write("[No text extracted]\n")
            f.write("\n")
        f.write("--- Media files ---\n")
        for name in media_names:
            f.write(f"{name}\n")

    print(OUT_PATH)


if __name__ == "__main__":
    main()
