from setuptools import setup
import importlib.util

SOFA_MODULE = "Sofa"
assert importlib.util.find_spec(SOFA_MODULE), f"Could not find {SOFA_MODULE} module. \n Please install SOFA with the SofaPython3 plugin."

setup(
    name="sofa_zoo",
    version="0.0.1",
    description="Reinforcement Learning Code for Envs from the sofa_env Repository.",
    author="Paul Maria Scheikl",
    author_email="paul.scheikl@kit.edu",
    packages=["sofa_zoo"],
    install_requires=[
        "sofa_env",
        "stable-baselines3==2.2.1",
        "tensorboard",
        "wandb",
        "gitpython",
        "tqdm",
        "torchvision",
        "pytest",
        "moviepy",
    ],
    python_requires=">=3.8",
)
