from setuptools import setup,find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    requirements=[]

    with open(file_path,"r") as f:
        requirements=f.readlines()
    
    requirements=[req.replace("\n","") for req in requirements]

    return requirements

setup(
    name="src",
    version="0.0.1",
    author="Anjaneyulu",
    author_email="baikanianji@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
