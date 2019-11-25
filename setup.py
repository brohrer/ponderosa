import setuptools as st

with open("README.md", "r") as fh:
    long_description = fh.read()

st.setup(
    name='ponderosa',
    version='1',
    description='A hyperparameter optimizer',
    url='http://github.com/brohrer/ponderosa',
    download_url='https://github.com/brohrer/ponderosa/tags/',
    author='Brandon Rohrer',
    author_email='brohrer@gmail.com',
    license='MIT',
    install_requires=[
        'matplotlib',
        'numpy',
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={
        "": [
            "README.md",
            "LICENSE",
        ],
        "ponderosa": [],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
