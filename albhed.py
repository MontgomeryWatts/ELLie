e_to_ab = {
  'a': 'y',
  'b': 'p',
  'c': 'l',
  'd': 't',
  'e': 'a',
  'f': 'v',
  'g': 'k',
  'h': 'r',
  'i': 'e',
  'j': 'z',
  'k': 'g',
  'l': 'm',
  'm': 's',
  'n': 'h',
  'o': 'u',
  'p': 'b',
  'q': 'x',
  'r': 'n',
  's': 'c',
  't': 'd',
	'u': 'i',
  'v': 'j',
  'w': 'f',
  'x': 'q',
  'y': 'o',
  'z': 'w'
}

ab_to_e = { v: k for k, v in e_to_ab.items() }

def translate_to_al_bhed(string: str) -> str:
  return translate(string, e_to_ab)

def translate_to_english(string: str) -> str:
  return translate(string, ab_to_e)

def translate(string: str, lang_map: dict) -> str:
	output = ""
	preserve_text = False
	for char in string:
		if preserve_text:
			if char == '*':
			  preserve_text = False
		elif char.lower() in lang_map.keys():
			if char.isupper():
				char = lang_map[char.lower()].upper()
			else:
				char = lang_map[char]
		else:
			if char == '*':
				preserve_text = True
		output += char
	return output