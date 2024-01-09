from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name='bioib',
    version='0.1.0',
    packages=find_packages(where="bioIB_py"),
    description="Biological Information Bottleneck",
    long_description=long_description,
    url="https://github.com/nitzanlab/bioib",
    author="Sima Dubnov",
    # This should be a valid email address corresponding to the author listed
    # above.
    author_email="serafima.dubnov@mail.huji.ac.il",
    classifiers = [
      "Development Status :: 5 - Production/Stable",
      "Intended Audience :: Science/Research",
      "Natural Language :: English",
      "Programming Language :: Python :: 3.8",
      "Programming Language :: Python :: 3.9",
      "Programming Language :: Python :: 3.10",
      "Operating System :: MacOS :: MacOS X",
      "Operating System :: Microsoft :: Windows",
      "Operating System :: POSIX :: Linux",
      "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.7, <4"
)
