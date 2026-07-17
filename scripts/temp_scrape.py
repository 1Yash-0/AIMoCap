import urllib.request
import re
html = urllib.request.urlopen('http://domedb.perception.cs.cmu.edu/dataset.html').read().decode('utf-8')
seqs = set(re.findall(r'href=[\'"]([0-9]{6}_[a-zA-Z0-9_]+)', html))
print('Sequences found:')
for s in sorted(seqs):
    print(s)
