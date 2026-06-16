import urllib.request, json
data = json.loads(urllib.request.urlopen('http://127.0.0.1:9000/openapi.json').read())
for path in sorted(data.get('paths', {}).keys()):
    print(path)
