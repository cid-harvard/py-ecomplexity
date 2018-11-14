from setuptools import setup, find_packages

setup(name='py-ecomplexity',
      version='0.1',
      description='Package to calculate economic complexity and associated variables',
      url='https://github.com/cid-harvard/py-ecomplexity',
      author='Shreyas Gadgin Matha',
      author_email='shreyas_gadgin_matha@hks.harvard.edu',
      license='MIT',
      packages=find_packages(),
      keywords="pandas python networks economics complexity",
      install_requires=[
          'pandas',
          'numpy'
      ],
      zip_safe=False)
