# Physics-informed mammogram synthesis (U24_synthesis DDPM) + quantum classifier study

Thesis project, September-October 2026. **Start with `HANDOVER.ipynb`** - full context, every study, every result,
how to run, what is left. Links to every input: `LINKS.md`. Prompt for an AI assistant: `PROMPT_FOR_AI_ASSISTANT.md`.

Base generative model: **U24_synthesis** - Linardos, Thakur, Gastounioti, Makris & Bakas, "Evaluating the utility of
AI-generated mammography images in breast cancer analysis", Scientific Reports 2026. Not our work; we build on it.

* `notebooks/` - Kaggle notebooks, each self-contained (code embedded). Upload to Kaggle, add the inputs listed in its first cell, run.
* `src/` - the same code as .py files; `builders/` regenerate the notebooks from `src/`.
* `preregistrations/`, `reports/`, `figures/` - rules written before each run, final reports, figures.
* `tests/test_qsim.py` - checks the quantum simulator against dense matrices (numpy only).

Datasets are NOT included (CBIS-DDSM and CMMD are on Kaggle/TCIA under their own licences).
