from setuptools import find_packages, setup

package_name = 'casea'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kris',
    maintainer_email='kris@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'ta_node = casea.ta_node:main',
        'keyboard_node = casea.keyboard_node:main',
        'ps5_node = casea.ps5_node:main'
        ],
    },
)
