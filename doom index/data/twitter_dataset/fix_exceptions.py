import os
import re

files = [
    "enhanced_scraper.py",
    "twitter_scraper_v10.py",
    "uc_login.py",
    "test_search.py",
    "enhanced_scraper_v5.py",
    "playwright_login.py",
    "graphql_scraper_v8.py",
    "enhanced_scraper_v6_fixed.py",
    "enhanced_scraper_v4.py",
    "graphql_scraper_v9.py",
    "playwright_login_v2.py",
    "enhanced_scraper_v7.py",
    "scraper_with_cookies.py",
    "enhanced_scraper_v3.py",
    "enhanced_scraper_v2.py",
    "selenium_login.py",
    "run_twitter_scraper.py",
    "enhanced_scraper_v6.py",
    "uc_auth.py",
    "get_twitter_auth.py",
]

base_dir = "/Users/ajay/Downloads/doom-index/doom index/data/twitter_dataset/"

def determine_exception(text, line_idx):
    # look backwards 10 lines
    start = max(0, line_idx - 15)
    context = "".join(text[start:line_idx])
    
    if "WebDriverWait" in context or "until(" in context:
        return "except TimeoutException as e:", "from selenium.common.exceptions import TimeoutException\n"
    if "find_element" in context or "css_selector" in context.lower():
        return "except NoSuchElementException as e:", "from selenium.common.exceptions import NoSuchElementException\n"
    if "json.load" in context or "json.dump" in context:
        return "except json.JSONDecodeError as e:", ""
    if "session.get" in context or "httpx" in context or "response" in context:
        return "except httpx.RequestError as e:", "import httpx\n"
    if "client." in context or "twikit" in context:
        return "except Exception as e: # twikit error", ""
    if "driver." in context:
        return "except WebDriverException as e:", "from selenium.common.exceptions import WebDriverException\n"
    if "open(" in context or "read(" in context or "write(" in context:
        return "except IOError as e:", ""
    
    return "except RuntimeError as e:", ""

audit_log = ["# Batch 6 Audit\n"]

for filename in files:
    filepath = os.path.join(base_dir, filename)
    if not os.path.exists(filepath):
        print(f"Skipping {filename}, does not exist.")
        continue
        
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    imports_to_add = set()
    modifications = 0
    
    for i, line in enumerate(lines):
        if "except Exception as e:" in line:
            indent = line[:len(line) - len(line.lstrip())]
            exc_line, imp = determine_exception(lines, i)
            new_lines.append(indent + exc_line + "\n")
            if imp:
                imports_to_add.add(imp)
            modifications += 1
            audit_log.append(f"- Fixed exception in `{filename}` at line {i+1}: replaced with `{exc_line.strip()}`")
        else:
            new_lines.append(line)
            
    if modifications > 0:
        # add imports at top
        final_lines = []
        imports_added = False
        for line in new_lines:
            final_lines.append(line)
            if not imports_added and (line.startswith("import ") or line.startswith("from ")):
                # just put them after the first import
                for imp in imports_to_add:
                    if imp not in "".join(new_lines):
                        final_lines.append(imp)
                imports_added = True
                
        with open(filepath, 'w') as f:
            f.writelines(final_lines)
            
        audit_log.append(f"-> Updated {filename} ({modifications} fixes)\n")

with open("/Users/ajay/.gemini/antigravity/brain/76efcab7-ec1c-4111-8660-e6e8b0e6c2eb/Batch_6_Audit.md", "w") as f:
    f.write("\n".join(audit_log))

print("Done")
