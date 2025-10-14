from setuptools import setup, find_packages
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setup(
    name='my_investment-package',
    version='1.0.0',
    packages=find_packages(),
    description='Descricao da sua lib my_investment',
    author='Matheus',
    author_email='matheus@example.com',
    url='https://github.com/tadrianonet/my_investment',  
    license='MIT',  
    long_description=long_description,
    long_description_content_type='text/markdown'
)
