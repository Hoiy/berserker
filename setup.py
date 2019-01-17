from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = 'berserker',
    version = '0.1',
    author = 'Hoi',
    author_email = 'hoiy927@gmail.com',
    description='Berserker - BERt chineSE toKenizER',
    long_description=long_description,
    install_requires = ['requests', 'six', 'tqdm'],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
