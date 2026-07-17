import urllib.request
import re

url = "http://domedb.perception.cs.cmu.edu/dataset.html"
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req)
    html = resp.read().decode('utf-8')
    links = set(re.findall(r'href=[\'\"]?([^\'\" >]+)', html))
    for link in sorted(links):
        if 'sync' in link.lower() or 'tar' in link.lower() or 'json' in link.lower() or 'time' in link.lower():
            print(link)
except Exception as e:
    print("Error:", e)
