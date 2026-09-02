import urllib.request
import json

url = "https://api.github.com/repos/killer1panda/prsi/actions/runs/33619397975"
req = urllib.request.Request(url)
# No auth provided, so might get rate limited or 404 if private
try:
    with urllib.request.urlopen(req) as response:
        print(response.getcode())
except Exception as e:
    print(e)
