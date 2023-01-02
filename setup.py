import setuptools

with open("requirements.txt") as f:
    required = f.read().splitlines()

if __name__ == "__main__":
    setuptools.setup(
        package_data={"grandmaster": ["py.typed"]},
        install_requires=required,
    )
