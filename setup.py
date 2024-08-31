from setuptools import setup, find_packages

setup(
    name='fpnn',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'matplotlib==3.9.2',
        'numpy==1.24.3',
        'pandas==2.2.2',
        'scikit-learn==1.5.0',
        'seaborn==0.13.2',
        'tensorflow==2.13.1',
    ],
    author='Mauricio Souza',
    author_email='m241948@dac.unicamp.br',
    description='This project consists in the use of feed forward neural networks to predict the flash point of biofuels',
    url='https://github.com/ThermoPhase-FCSRG/fp-neural-net',
    classifiers=[
        'Programming Language :: Python :: 3.10',  
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)