# CLAUDE.md - Agent-Memory-Paper-List

## Description

Curated academic paper list accompanying "Memory in the Age of AI Agents: A Survey" (arXiv:2512.13564). Fork of Shichun-Liu/Agent-Memory-Paper-List. Documentation-only knowledge base — no executable code.

## What This Is

- 196 research papers organized in a 3x3 taxonomy matrix
- **Memory Forms** (columns): Token-level, Parametric, Latent
- **Memory Functions** (rows): Factual, Experiential, Working
- 99 PDFs in PAPERS/ (79 valid, 20 zero-byte placeholders)
- 2 taxonomy diagrams in assets/

## Structure

```
Agent-Memory-Paper-List/
├── README.md          # Main paper catalog (296 lines, 32 KB)
├── PAPERS/            # Downloaded PDFs by arXiv ID (339 MB)
├── assets/            # main.png (taxonomy), concept.png (comparison)
└── LICENSE            # MIT
```

## Key Numbers

- Total papers cataloged: 196
- PDFs downloaded: 79 valid + 20 empty placeholders
- Commits: 29 (Dec 2025 - Jan 2026)
- Repository size: 364 MB (mostly PDFs)

## Known Issues

- 20 zero-byte PDFs need downloading (2511.x - 2601.x range)
- README references 196 papers but only 99 PDFs exist locally
- Duplicate entry at line 64 (same as line 59)
- No upstream remote configured (only origin=josem4pro)
- assets/.DS_Store should be in .gitignore

## Relevance to Workstation IA

This taxonomy (Factual/Experiential/Working x Token/Parametric/Latent) provides theoretical framework for evaluating claude-mem's memory model. Key papers: Mem0, A-MEM, MemGPT, HippoRAG, Memory-R1.
