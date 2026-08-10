# Contributing

Please keep published input tables immutable. Add new experiment runs under a seed-scoped directory and include the command line, Git commit, software environment, hardware description, and random seed.

Before submitting a change:

```bash
python -c "from pathlib import Path; [compile(p.read_text(encoding='utf-8'), str(p), 'exec') for p in Path('multiseed/scripts').glob('*.py')]"
python analysis/summarize_results.py
```

Do not commit model checkpoints, local virtual environments, credentials, or machine-specific absolute paths. Changes to energy constants or system boundaries must update the documentation and regenerate all affected result tables and figures.

