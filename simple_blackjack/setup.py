import setuptools
from setuptools import setup
setup(name='simple_blackjack')

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Simple-Blackjack", # Replace with your own username
    version="0.0.1",
    author="Eric Van Cleave",
    author_email="ericvc2@gmail.com",
    description="A simple blackjack game for your terminal window.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ericvc/Simple-Blackjack",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)
