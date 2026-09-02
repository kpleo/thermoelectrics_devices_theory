.PHONY: reproduce test check

reproduce:
	python scripts/reproduce_results.py --from-processed

test: reproduce
	python -m pytest -q

check: test
