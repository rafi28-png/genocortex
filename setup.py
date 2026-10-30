"""Build optional native C accelerator (fpc._fast) with graceful fallback."""
from __future__ import annotations

import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.errors import CCompilerError, ExecError, PlatformError


class OptionalBuildExt(build_ext):
    """Allow pure-Python installation if no C compiler is present on target."""

    def run(self) -> None:
        try:
            super().run()
        except (PlatformError, ExecError, CCompilerError, OSError) as exc:
            print(f"[genocortex] warning: skipping C accelerator build ({exc}); falling back to NumPy.")

    def build_extension(self, ext: Extension) -> None:
        try:
            super().build_extension(ext)
        except (PlatformError, ExecError, CCompilerError, OSError) as exc:
            print(f"[genocortex] warning: could not compile {ext.name} ({exc}); falling back to NumPy.")


if sys.platform == "win32":
    compile_args = ["/O2"]
    link_args: list[str] = []
else:
    compile_args = ["-O3", "-std=c11"]
    link_args = ["-lm"]

setup(
    ext_modules=[
        Extension(
            "fpc._fast",
            sources=["fpc/_fast.c"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            optional=True,
        )
    ],
    cmdclass={"build_ext": OptionalBuildExt},
)
