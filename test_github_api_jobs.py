import urllib.request
import json

url = "https://api.github.com/repos/killer1panda/prsi/actions/runs/33619397975/jobs"
req = urllib.request.Request(url)
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        for job in data.get("jobs", []):
            if job["name"] == "github-advanced-security":
                print("Found job:", job["id"])
                print(job)
except Exception as e:
    print(e)
