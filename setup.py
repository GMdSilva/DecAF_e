from setuptools import setup, find_packages

setup(
    name="decaf_e",
    version="0.1.0",
    author="Gabriel Monteiro da Silva",
    author_email="gabriel.monteiro233@gmail.com",
    description="DECoding AlphaFold2 Ensembles with Attention",
    long_description=open('README').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/GMdSilva/DECAfold-E",
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'scikit-learn',
        'scipy',
        'numpy',
        'matplotlib',
        'pandas'
    ]
)