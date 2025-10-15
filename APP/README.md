Project reorganized into packages.

How to run the UI from the `ui` package:

- From the project root (c:\Users\<you>\Desktop\dev\APP):

  python -m ui

- Or, to run specific file directly:

  python ui.py

Notes:
- A new `core` package contains shared `utils` and `models`.
- `yolo.py` contains helpers used by the UI.
- `yolo_visualizeV4.py` remains for backward compatibility but imports core utilities.
