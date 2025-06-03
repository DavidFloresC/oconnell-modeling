# setup.py

from setuptools import setup, find_packages

setup(
    name='oconnell_modeling',
    version='0.1.0',
    description="Phenomenological modeling of the O'Connell effect in eclipsing binaries",
    author='D. Flores Cabrera et al.',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'astropy',
        'lmfit',
        'symfit'
    ],
    entry_points={
        'console_scripts': [
            'oconnell-modeling=cli:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
