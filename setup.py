from setuptools import setup, find_packages

with open("README.md", 'r', encoding="utf8") as handle:
    description = handle.read()

setup(
    name='prodfs',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.15.0',
        'scipy>=1.8.0',
        'matplotlib>=3.6'
    ],
    long_description=description,
    long_description_content_type="text/markdown",
)
