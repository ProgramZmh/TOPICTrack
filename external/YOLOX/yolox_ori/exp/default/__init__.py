

import importlib
import sys
from pathlib import Path

_EXP_PATH = Path(__file__).resolve().parent.parent.parent.parent / "exps" / "default"

if _EXP_PATH.is_dir():
    

    class _ExpFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if not name.startswith("yolox.exp.default"):
                return
            project_name = name.split(".")[-1] + ".py"
            target_file = _EXP_PATH / project_name
            if not target_file.is_file():
                return
            return importlib.util.spec_from_file_location(name, target_file)

    sys.meta_path.append(_ExpFinder())
