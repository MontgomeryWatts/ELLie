from io import StringIO
import asyncio
from concurrent.futures import ProcessPoolExecutor

from tatsu.util import generic_main

from boi.parser import BoiParser
from boi.semantics import BoiSemantics

__EXECUTOR = None

def try_run_src(src):
    start = 'start'
    parser = BoiParser()

    try:
        parsed = parser.parse(src, rule_name=start, filename="", semantics=BoiSemantics(), **{})
    except Exception as e:
        return f"Encountered the following error after attempted parse:\n{str(e)}"

    sout = StringIO()
    try:
        parsed.run(stdout=sout)
        return sout.getvalue()
    except Exception as e:
        return f"{sout.getvalue()}\nEncountered the following error while running:\n{str(e)}"

def run(src):
    start = 'start'
    parser = BoiParser()

    parsed = parser.parse(src, rule_name=start, filename="", semantics=BoiSemantics(), colorize=True)
    sout = StringIO()
    parsed.run(stdout=sout)

    return sout.getvalue()


async def async_run(src):
    global __EXECUTOR
        
    if __EXECUTOR is None:
        __EXECUTOR = ProcessPoolExecutor(max_workers=3)


    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(__EXECUTOR, try_run_src, src) 
    return res