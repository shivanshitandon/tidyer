from glob import glob
from setuptools import find_packages, setup

package_name = 'tidyer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy', 'opencv-python'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'process_pointcloud = tidyer.perception.process_pointcloud:main',
            'tidyer_tf = tidyer.planning.static_tf_transform:main',
            'tidyer_pick_place = tidyer.planning.main:main',
            'tidyer_ik = tidyer.planning.ik:main',
        ],
    },
)
