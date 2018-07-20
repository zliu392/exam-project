from setuptools import setup, find_packages

setup(
    name='experimentasmarket',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='GNU',
    description='A python package for the EXAM algorithm.',
    long_description=open('README.md').read(),
    install_requires=['numpy', 'scipy', 'math', 'collections', 'random', 'copy', 'matplotlib', 'timeit', 'pandas', 'decimal', 'time'],
    url='https://github.com/zliu392/exam-project',
    author='Yusuke Narita',
    author_email='yusuke.narita@yale.edu'
)
