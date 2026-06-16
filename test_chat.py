import urllib.request, json
body = json.dumps({"messages": [{"role": "user", "content": "GH05T3 who are you?"}]}).encode()
req  = urllib.request.Request("http://127.0.0.1:7000/chat", data=body,
    headers={"Content-Type": "application/json"}, method="POST")
resp = urllib.request.urlopen(req, timeout=90)
data = json.loads(resp.read())
print("REPLY:", data.get("reply",""))
print("TIME: ", data.get("elapsed",""),"s")
