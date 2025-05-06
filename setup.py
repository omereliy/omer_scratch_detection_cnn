from setuptools import setup, find_packages

setup(
    name="wafer_scratch_detection",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.12.1",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "scipy",
        "scikit-image",
    ],
)