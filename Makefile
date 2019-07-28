boi_tests:
	python -m pytest src/boi/test -s

run:
	python src

grammar:
	tatsu --generate-parser src/boi/tatsu_grammar -o src/boi/parser.py
