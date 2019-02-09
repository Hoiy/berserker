from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = 'basaka',
    version = '0.2.1',
    author = 'Hoi',
    author_email = 'hoiy927@gmail.com',
    description='Berserker - BERt chineSE woRd toKenizER',
    long_description=long_description,
    install_requires = ['requests', 'six', 'tqdm', 'tensorflow>=1.12.0'],
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
