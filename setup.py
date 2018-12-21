from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='ecomplexity',
      version='0.3',
      description='Package to calculate economic complexity and associated variables',
      long_description=readme(),
      url='https://github.com/cid-harvard/py-ecomplexity',
      author='Shreyas Gadgin Matha',
      author_email='shreyas_gadgin_matha@hks.harvard.edu',
      license='MIT',
      packages=find_packages(),
      keywords="pandas python networks economics complexity",
      python_requires='>=3',
      install_requires=[
          'pandas >0.23.0',
          'numpy >1.15.0'
      ],
      zip_safe=False)
