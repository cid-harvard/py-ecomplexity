from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        return f.read()

setup(name='ecomplexity',
      version='0.5.2',
      description='Package to calculate economic complexity and associated variables',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/cid-harvard/py-ecomplexity',
      author='Shreyas Gadgin Matha',
      author_email='shreyas.gm61@gmail.com',
      license='MIT',
      packages=find_packages(),
      keywords="pandas python networks economics complexity",
      python_requires='>=3',
      install_requires=[
          'pandas >0.23.0',
          'numpy >1.22.0',
          'scikit-learn >1.0.0'
      ],
      zip_safe=False,
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ])
