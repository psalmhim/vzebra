from setuptools import setup, find_packages

setup(
    name="vzlab",
    version="0.1.0",
    description="Virtual Laboratory: multi-species hierarchical brain simulation platform",
    packages=find_packages(include=["vzlab", "vzlab.*"]),
    python_requires=">=3.10",
    install_requires=["numpy>=1.24"],
    extras_require={
        "viz": ["matplotlib>=3.7"],
        "full": ["torch>=2.0", "gymnasium>=0.29", "matplotlib>=3.7"],
    },
    entry_points={
        "console_scripts": [
            "vzlab=vzlab.cli:main",
            "vzlab-web=vzlab.web.server:main",
        ],
    },
)
