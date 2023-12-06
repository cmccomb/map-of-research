#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='map_of_research',
    version='0.1.0',
    description='Mapping research for lists of google scholar profiles',
    author='Chris McComb',
    author_email='ccm@cmu.edu',
    url='https://cmccomb.com',
    install_requires=["pandas", "scholarly", "plotly", "scikit-learn", "numpy", "sentence_transformers", "matplotlib"],
    entry_points={
        'console_scripts': [
            'scrape_faculty_data=map_of_research:scrape_faculty_data',
            'visualize_faculty_data=map_of_research:visualize_faculty_data'
        ]
    },
    packages=find_packages(),
)
