from setuptools import setup

setup(
    name = 'eztorch',
    version = '0.1.0',
    description = 'Train pytorch models in one line',
    license='MIT',
    packages = ['eztorch'],
    install_requires = ['pytorch'],
)