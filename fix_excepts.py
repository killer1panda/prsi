import os, re
for root, dirs, files in os.walk('.'):
    if any(x in root for x in ['node_modules', '.git', '__pycache__', 'venv', '.next']): continue
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Replace 'except:' with 'except Exception as e:' (handling whitespace)
            new_content = re.sub(r'^[ \t]*except[ \t]*:[ \t]*(?:#.*)?$', lambda m: m.group(0).replace('except:', 'except Exception as e:'), content, flags=re.MULTILINE)
            # Replace 'except :'
            new_content = re.sub(r'^[ \t]*except\s+:[ \t]*(?:#.*)?$', lambda m: m.group(0).replace('except :', 'except Exception as e:'), new_content, flags=re.MULTILINE)
            if new_content != content:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Fixed bare excepts in {path}")
