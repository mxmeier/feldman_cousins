from setuptools import setup

setup(name='feldman_cousins',
      version='0.1',
      description='Calculate Feldman Cousins confidence intervals.',
      url='http://github.com/mxmeier/feldman_cousins',
      author='Maximilian Meier',
      author_email='maximilian.meier@udo.edu',
      license='MIT',
      packages=['feldman_cousins'],
      install_requires=[
        'scipy',
        'numpy',
        'matplotlib',
        'click'],
      zip_safe=False)
