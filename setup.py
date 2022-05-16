from setuptools import setup, find_packages

setup(
    name = 'eztorch',
    version = '0.1.0',
    description = 'Train pytorch models in one line',
    license='MIT',
    packages = find_packages(
        where='eztorch'
    ),
    python_requires='>=3.7',
    include_package_data=True
)