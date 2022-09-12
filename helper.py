#  función para eliminar las tildes introducidas --> meterlo en helper
def normalize(s):
  replacements = (
      ("á", "a"),
      ("é", "e"),
      ("í", "i"),
      ("ó", "o"),
      ("ú", "u"),
  )
  for a, b in replacements:
      s = s.replace(a, b).replace(a.upper(), b.upper())
  return s
  