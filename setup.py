"""Setup for yijing-transformer package."""

from setuptools import setup, find_packages

with open("yijing-transformer-concept.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="yijing-transformer",
    version="0.50.0",
    author="YiJing-Transformer Team",
    description="Transformer with hypercube geometry inductive bias (Yi Jing hexagrams)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/svend4/pro2",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "data": ["datasets>=2.14.0", "sentencepiece>=0.2.0"],
        "viz": ["matplotlib>=3.7.0"],
        "tracking": ["wandb>=0.16.0", "tensorboard>=2.13.0"],
        "dev": ["pytest"],
        "all": ["datasets>=2.14.0", "sentencepiece>=0.2.0", "matplotlib>=3.7.0", "wandb>=0.16.0", "tensorboard>=2.13.0", "pytest"],
    },
    entry_points={
        "console_scripts": [
            "yijing-train=yijing_transformer.scripts.train_model:main",
            "yijing-wikitext=yijing_transformer.scripts.wikitext_train:main",
            "yijing-extensions=yijing_transformer.scripts.run_all_extensions:main",
            "yijing-downstream=yijing_transformer.scripts.downstream_finetune:run_experiment",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="transformer hypercube yijing hexagram geometry quantization nlp",
)
