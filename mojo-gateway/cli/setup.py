"""
EdgeLLM CLI Setup.

Install with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="edgellm",
    version="0.1.0",
    description="Fine-tune, optimize, and deploy LLMs to edge devices",
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
    author="EdgeLLM Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.0",
        "rich>=10.0.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "finetune": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "peft>=0.4.0",
            "bitsandbytes>=0.40.0",
            "datasets>=2.0.0",
            "accelerate>=0.20.0",
        ],
        "serve": [
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
            "pydantic>=2.0.0",
        ],
        "all": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "peft>=0.4.0",
            "bitsandbytes>=0.40.0",
            "datasets>=2.0.0",
            "accelerate>=0.20.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
            "pydantic>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "edgellm=edgellm.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
