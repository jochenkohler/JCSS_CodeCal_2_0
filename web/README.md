# The interface as a static web page

`index.html` runs `app.py` in the browser with [stlite](https://github.com/whitphx/stlite)
(Streamlit on Pyodide / WebAssembly): no server, no installation, nothing leaves the
user's machine. It needs one file next to it, `codecal_app.zip`, holding `app.py`,
`.streamlit/config.toml` and the `codecal/` package:

    make web            # builds web/codecal_app.zip

The GitHub Actions workflow (`.github/workflows/book.yml`) builds the zip and publishes
`web/` as `app/` next to the Jupyter Book on GitHub Pages, so the interface is at
`https://<user>.github.io/<repo>/app/`.

To try it locally: `make web && cd web && python -m http.server 8000`, then open
http://localhost:8000/ (the stlite and Pyodide files come from jsDelivr).

Limits of the browser version: the Monte-Carlo simulation of the characteristic loads
is capped at 1 million draws, calibrations run several times slower than installed,
and files you upload stay in the browser's memory. Streamlit Community Cloud
(https://share.streamlit.io, deploy from this repository, entry point `app.py`) is the
full-speed hosted alternative.
