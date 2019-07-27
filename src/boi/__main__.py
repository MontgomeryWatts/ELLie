import sys

from tatsu.util import generic_main

from boi.parser import BoiParser
from boi.semantics import BoiSemantics

def main(filename, start=None, **kwargs):
    if start is None:
        start = 'start'
    if not filename or filename == '-':
        text = sys.stdin.read()
    else:
        with open(filename) as f:
            text = f.read()
    parser = BoiParser()
    return parser.parse(text, rule_name=start, filename=filename, semantics=BoiSemantics(), **kwargs)


if __name__ == '__main__':
    import json
    from tatsu.util import asjson

    ast = generic_main(main, BoiParser, name='Boi')

    ast.run()