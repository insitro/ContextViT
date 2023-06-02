import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="contextvit",
    py_modules=["contextvit"],
    version="1.0",
    description="Contextual Vision Transformers for Robust Representation Learning",
    author="Insitro",
    packages = find_packages() + ['contextvit/config'],
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    # extras_require={'dev': ['pytest']},
)
