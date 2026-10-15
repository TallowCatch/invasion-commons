# DeepMind Lab2D repository (the engine under Melting Pot)

- **Authors:** DeepMind (Beattie et al.; whitepaper arXiv:2011.07027)
- **Year:** 2020 (whitepaper); repo latest commit 2 Oct 2026
- **Identifier:** github.com/google-deepmind/lab2d, latest commit `f6de2d6e40e476093b6413cd496137ad8fd8d3d5` (read via GitHub API; README and LICENSE fetched, repo not cloned)
- **URL:** https://github.com/google-deepmind/lab2d
- **Source type:** primary-code / docs
- **Date accessed:** 14 Oct 2026

## Verbatim quotes (README.md)

1. > "A learning environment written in C++ and Lua for the creation of grid worlds."
2. > "The environments are "grid worlds", which are defined with a combination of simple text-based maps for the layout of the world, and Lua code for its behaviour."
3. > "Machine learning agents interact with these environments through one of two APIs, the Python [`dm_env`](https://github.com/deepmind/dm_env) API or a custom C API"
4. > "`dmlab2d` is distributed as pre-built wheels for Linux and macOS. If there is no appropriate wheel for your platform, you will need to build it from source."
5. LICENSE first line: "Apache License Version 2.0, January 2004"

## Relevance
This confirms the engine stack: C++ core, Lua game logic, Python dm_env API, Apache-2.0. It runs on CPU one environment at a time, which is why SocialJax-style JAX ports report large speedups (covered by the SocialJax sub-agent).
