"""
Setup script for CondensedTSF project
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="condensed-tsf",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="CondensedTSF: Data Condensation for Time Series Forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/condensed-tsf",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "condensed-train=scripts.train:main",
            "condensed-test=scripts.test_model:main",
            "condensed-pretrain=scripts.pretrain_encoder:main",
            "condensed-condense=scripts.get_condensed_dataset:main",
        ],
    },
)
