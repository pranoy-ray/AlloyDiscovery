from setuptools import setup, find_packages

setup(
    name='alloy_discovery',
    version='0.1.0',
    description='Universal electronic manifolds for extrapolative alloy discovery',
    author='Pranoy Ray et al.',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'h5py',
        'torch',
        'gpytorch',
        'botorch',
        'ase',
        'pyvista',
        'matplotlib'
    ],
)
