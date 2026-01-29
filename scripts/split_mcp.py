import re
import sys
from pathlib import Path

src_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
text = src_path.read_text(encoding='utf-8', errors='replace')
parts = re.split(r'(?m)^# (.+)$', text)

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
    return s[:80] or 'section'

wanted = {
    'architecture overview': 'architecture_overview',
    'lifecycle': 'lifecycle',
    'transports': 'transports',
    'resources': 'resources',
    'tools': 'tools',
    'roots': 'roots',
    'elicitation': 'elicitation',
    'sampling': 'sampling',
    'logging': 'logging',
    'pagination': 'pagination',
    'cancellation': 'cancellation',
    'progress': 'progress',
    'prompts': 'prompts',
    'schema': 'schema_reference',
    'security': 'security_best_practices',
    'authorization': 'authorization',
}

sections = []
# parts: [preamble, title1, body1, title2, body2, ...]
for i in range(1, len(parts), 2):
    title = parts[i].strip()
    body = parts[i+1].strip() if i+1 < len(parts) else ''
    key = title.strip().lower()
    # match desired sections loosely
    match = None
    for k in wanted:
        if k in key:
            match = wanted[k]
            break
    if match:
        sections.append((match, title, body))

# write files
for idx, (stem, title, body) in enumerate(sections, start=1):
    path = out_dir / f"{idx:02d}_{stem}.md"
    path.write_text(f"# {title}\n\n{body}\n", encoding='utf-8')

(out_dir / "README.md").write_text(
    "# MCP curated snapshot\n\n"
    "Generated from modelcontextprotocol.io/llms-full.txt and split into topic files for RAG.\n",
    encoding='utf-8'
)
print(f"Wrote {len(sections)} MCP files to {out_dir}")
