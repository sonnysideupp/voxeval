"""The README's opening example is the library's thesis, so its printed numbers must be real."""

import contextlib
import io
import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_quickstart_prints_what_it_claims() -> None:
    code = re.search(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert code is not None
    source = code.group(1)
    claimed = re.findall(r"print\(.*\)\s*#\s*([\d.]+)", source)
    claimed_exit = re.search(r"sys\.exit\(.*\)\s*#\s*(\d+)", source)
    assert claimed and claimed_exit is not None

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            exec(compile(source, str(README), "exec"), {"__name__": "readme"})
        except SystemExit as stop:
            exit_code = stop.code
        else:
            exit_code = None

    assert [float(v) for v in out.getvalue().split()] == [float(v) for v in claimed]
    assert exit_code == int(claimed_exit.group(1))
