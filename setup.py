#!/usr/bin/env python
# This file is for backward compatibility with older pip versions
# The main configuration is in pyproject.toml

from setuptools import setup

if __name__ == "__main__":
    setup(
        name="stonefish",
        packages=["stonefish", "stonefish.eval", "stonefish.train"],
    )
