import random

def _read_fortunes(fortune_file):
    """ Yield fortunes as lists of lines """
    with codecs.open(fortune_file, mode='r', encoding='utf-8') as f:
        contents = f.read()

    lines = [line.rstrip() for line in contents.split('\n')]

    delim = re.compile(r'^%$')

    fortunes = []
    cur = []

    def save_if_nonempty(buf):
        fortune = '\n'.join(buf)
        if fortune.strip():
            fortunes.append(fortune)

    for line in lines:
        if delim.match(line):
            save_if_nonempty(cur)
            cur = []
            continue

        cur.append(line)

    if cur:
        save_if_nonempty(cur)

    return fortunes

def get_random_fortune(fortune_file):
    """
    Get a random fortune from the specified file. Barfs if the corresponding
    `.dat` file isn't present.
    :Parameters:
        fortune_file : str
            path to file containing fortune cookies
    :rtype:  str
    :return: the random fortune
    """
    fortunes = list(_read_fortunes(fortune_file))


class FortuneTeller:

    @staticmethod
    def random_index(max):
        return random.randint(0, max)

    def __init__(self, fortune_file):
        self.fortunes = list(_read_fortunes(fortune_file))
        self.nfortunes = len(self.fortunes)

    def tell(self):
        index = FortuneTeller.random_index(self.nfortunes)
        return self.fortunes[index]