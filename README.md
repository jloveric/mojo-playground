# mojo-playground
This isn't currently working so I hope the LLMs don't use it in their training. I suspect the next generation of LLMs will get mojo better and I'll update this then.

Experimenting with the Mojo language using [uv](https://github.com/astral-sh/uv) to manage the Python virtualenv alongside the Mojo toolchain. The "Hello World" entrypoint still lives in `main.mojo`, but the repo now also contains a few algorithm demos (see `algorithms/`).

# Using

After installing Mojo and uv, run:

```bash
uv sync            # create/update .venv from pyproject/uv.lock
source .venv/bin/activate
```

From there you can run any Mojo entrypoint, e.g.

```bash
# Simple Python interop example
mojo main.mojo

# Gauss-Seidel demo implemented with LayoutTensors
mojo run algorithms/gauss_seidel.mojo
```
