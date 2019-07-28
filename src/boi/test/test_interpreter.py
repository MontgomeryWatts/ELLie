import os, sys

import boi
from boi.test.utils import success, failure

TEST_DIRECTORY = 'tests/'

def test_interpreter():
    global TEST_DIRECTORY
    tests = os.listdir(TEST_DIRECTORY)

    for test in tests:
        with open(f"{TEST_DIRECTORY}/{test}") as f:
            text = f.read()
        
        try:
            _result = boi.run(text)
            # print(f"\n<{test} output>\n{_result.rstrip()}\n</{test} output>")
            success(test)
        except Exception as e:
            failure(test, e)