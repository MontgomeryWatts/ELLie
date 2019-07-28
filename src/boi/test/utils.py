import traceback

def success(test_name):
    print(f"\n ╠═ {test_name + ' test':34} :: Okay", end='')

def failure(test_name, reason):
    print(f"\n ╠═ {test_name + ' test':34} :: FAILED with reason:\n{str(reason)}", end='')
    traceback.print_exc()