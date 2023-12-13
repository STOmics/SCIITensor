from  setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name='SCIITensor',
    version='1.0.0',
    author='Huanghuaqiang',
    author_email='huanghuaqiang@genomics.cn',
    description=('decipher tumor microenvironment via spatial cell-cell communication'),
    long_description=long_description,
    license='MIT License',
    keywords='TME',
    url="https://github.com/STOmics/SCIITensor",

    packages=['SCIITensor'], #需要打包的目录列表

    include_package_data=True,
    platforms='any',
    #需要安装的依赖包
    install_requires = [
        'anndata>=0.8.0',
        'scanpy>=1.9.2',
        'squidpy>=1.2.2',
        'scipy>=1.9.1',
        'tensorly>=0.8.1',
        'scvelo>=0.2.5',
        'scikit-learn>=1.2.1',
        'torch>=2.1.1',
        'igraph>=0.10.4',
        'pycirclize>=1.1.0'
    ],
)