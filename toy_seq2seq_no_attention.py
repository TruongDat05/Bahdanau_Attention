"""Compatibility wrapper for the old single-file Level 2 script.

The project is now split into dataset.py, vocab.py, model.py, train.py,
translate.py, and evaluate.py. Run this file only if you still use the old
command name.
"""

from train import main


if __name__ == "__main__":
    main()
