from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='TS_Tell',
    version='0.1.0',
    description='TS-Tell - Time Series - Tell',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/thompsonml/TS-Tell',
    author='Matt Thompson',
    author_email='GoBucksFromVA@gmail.com',
    packages=find_packages(),
    install_requires=[
        #'dependency1',
        #'dependency2>=1.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)