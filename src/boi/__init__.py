from io import StringIO

from tatsu.util import generic_main

from boi.parser import BoiParser
from boi.semantics import BoiSemantics

def run(text, **kwargs):
    start = 'start'
    parser = BoiParser()
    parsed = parser.parse(text, rule_name=start, filename="", semantics=BoiSemantics(), **kwargs)
    sout = StringIO()
    try:
        parsed.run(stdout=sout)
        return sout.getvalue()
    except Exception as e:
        return f"{sout.getvalue()}\nEncountered the following error while running:\n{str(e)}"