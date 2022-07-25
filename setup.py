from distutils.core import setup
import glob

setup(
    name="stonefish",
    scripts=glob.glob("bin/*"),
    packages=["stonefish", "stonefish.eval"],
)
