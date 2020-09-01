
from setuptools import setup, find_packages
from distutils.command.install import INSTALL_SCHEMES


for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

setup(name='drl_surface_seg',
      version='0.0.1',
      description='Module for DRL on surface segregation systems',
      url='https://github.com/ulissigroup/surface-seg',
      author='Jun Yoon, Yuyang, Amir Barati Farimani, Zack Ulissi',
      author_email='zulissi@andrew.cmu.edu',
      license='GPL',
      platforms=[],
      packages=find_packages(),
      scripts=[],
      include_package_data=False,
      install_requires=['tensorflow==2.2.0', 
			'tensorforce==0.5.5',
			'ase>=3.19.1',
			'numpy',
			'matplotlib',
			'asap3',
            'seaborn',
			'sella @ git+https://github.com/zadorlab/sella.git',
            'amptorch'],
      long_description='''Module for implementing DRL in surface segregation studies.''',)
