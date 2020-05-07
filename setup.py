from setuptools import find_packages, setup

setup(
  name='deep_learning',
  version='0.1.0',
  author='fantianlong',
  url='https://github.com/RingZEROtlf/hit_deep_learning_2020',
  description='Pytorch Implementation for PRDL 2020 Labs for HIT',
  author_email='RingZEROtlf@outlook.com',
  package_data={
    '': [
      'configs/PRDL2020_lab2/*.yaml',
      'configs/PRDL2020_lab3/*.yaml',
      'configs/PRDL2020_lab3_aug/*.yaml',
      'configs/PRDL2020_lab3_resnet/*.yaml',
      'configs/PRDL2020_lab3_seblock/*.yaml',
      'configs/PRDL2020_lab3_vgg/*.yaml',
    ],
  },
  packages=find_packages(exclude=(
    'scripts',
    'tools',
  )),
  entry_points={
    'console_scripts': ['deep_learning=deep_learning.__main__:main'] 
  },
)
