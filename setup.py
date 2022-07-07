from setuptools import setup, find_packages

setup(
    name = 'eztorch',
    version = '0.1.1',
    description = 'Train pytorch models in one line',
    license='MIT',
    packages = find_packages(),
    python_requires='>=3.8',
    include_package_data=True
)
