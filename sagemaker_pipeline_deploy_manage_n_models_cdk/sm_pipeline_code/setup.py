import setuptools

required_packages = ["sagemaker", "boto3"]
setuptools.setup(
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=required_packages,
)
