from setuptools import find_packages,setup

setup(
    name='mcq_generator',
    version='0.0.1',
    author='shiva shankar',
    author_email='ji.shiva1998@Gmail.com',
    install_requires=["openai","langchain","streamlit","python-dotenv","PyPDF2"],
    packages=find_packages()
)