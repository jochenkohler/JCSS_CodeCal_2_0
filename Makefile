# JCSS CodeCal 2.0 — build targets
#   make test        run the test suite
#   make notebooks   re-execute the lessons and the documentation notebooks (≈ 15 min)
#   make book        build the Jupyter Book into _build/html
#   make web         build web/codecal_app.zip for the browser version of the interface
#   make site        book + web, assembled as GitHub Pages would serve them (_build/html, _build/html/app)

PY ?= python3

.PHONY: test notebooks book web site clean

test:
	$(PY) -m pytest -q

notebooks:
	cd lessons && bash build_notebooks.sh
	cd docs/eurocode_study && MPLBACKEND=Agg jupytext --to ipynb --execute --run-path . model.py -o model.ipynb
	cd docs/eurocode_study && MPLBACKEND=Agg jupytext --to ipynb --execute --run-path . results.py -o results.ipynb
	cd docs/manual && MPLBACKEND=Agg jupytext --to ipynb --execute --run-path . scripting.py -o scripting.ipynb

book:
	jupyter-book build .

web:
	rm -f web/codecal_app.zip
	zip -q -r web/codecal_app.zip app.py .streamlit/config.toml codecal -x '*__pycache__*' '*.pyc'
	@ls -la web/codecal_app.zip

site: book web
	mkdir -p _build/html/app
	cp web/index.html web/codecal_app.zip _build/html/app/
	@echo "site in _build/html (book) and _build/html/app (interface)"

clean:
	rm -rf _build web/codecal_app.zip
