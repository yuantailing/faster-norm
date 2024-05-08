from setuptools import setup


classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: GPU :: NVIDIA CUDA",
    "Programming Language :: Python :: 3",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Topic :: Software Development :: Libraries :: Python Modules",
]


with open("README.md", "r") as fp:
    long_description = fp.read()


setup(name="fast_norm_cuda",
      version="0.1.0",
      author="Tailing Yuan",
      author_email="yuantailing@gmail.com",
      url="https://github.com/yuantailing/fast-norm-cuda",
      tests_require=["pytest", "torch"],
      description="A fast, yet specialized, RMSNorm/LayerNorm implementation",
      long_description=long_description,
      license="MIT",
      classifiers=classifiers,
      python_requires=">=3.3",
      include_package_data=True,
      )
