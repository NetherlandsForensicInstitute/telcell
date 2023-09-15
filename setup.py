from pathlib import Path

from setuptools import find_packages, setup


long_description = (Path(__file__).parent / 'README.md').read_text()


dependencies = (
    'lir',
    'lrbenchmark >= 0.1.1',
    'numpy'
)


setup(
    name='telcell',
    version='0.0.2',
    author='Netherlands Forensics Institute',
    description='Calculating LRs for Collocated Tracks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=dependencies,
)