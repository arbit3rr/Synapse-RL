from setuptools import setup, find_packages

setup(
    name="synapse-rl",
    version="1.0.0",
    description="A clean, modular PyTorch library for deep reinforcement learning",
    author="Amirhossein Heydarian Ardakani",
    url="https://github.com/amirhosseinh77/Synapse-RL",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "gymnasium>=0.26.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "tensorboard>=2.10.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
