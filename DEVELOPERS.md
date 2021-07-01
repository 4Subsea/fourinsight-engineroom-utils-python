# Developers

## Set up dev environment
First, create a virtual environment:

```
python -m venv .venv
```
and activate:

```
.venv/scripts/activate
```

Then install the necessary packages for development:

```
pip install -r dev-requirements.txt
```

## Build source and wheel

```
python -m build --outdir ./build
```

## Build documentation

```
sphinx-build -W -b html ./docs ./build/docs
```

## Run tests

```
tox
```
