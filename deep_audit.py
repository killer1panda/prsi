import os
import ast
import re

def audit_repo():
    issues = []
    files_scanned = 0
    
    for root, dirs, files in os.walk('.'):
        if any(x in root for x in ['node_modules', '.git', '.next', '__pycache__', 'target', '.expo', 'venv']):
            continue
            
        for file in files:
            path = os.path.join(root, file)
            
            # Audit Python Files
            if file.endswith('.py'):
                files_scanned += 1
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # 1. Syntax Check
                    try:
                        tree = ast.parse(content)
                    except SyntaxError as e:
                        issues.append(f"[CRITICAL] Syntax Error in {path}: {e}")
                        continue
                        
                    # 2. AST Checks
                    for node in ast.walk(tree):
                        # Find bare excepts
                        if isinstance(node, ast.ExceptHandler) and node.type is None:
                            issues.append(f"[WARNING] Bare 'except:' clause in {path} at line {node.lineno} (swallows all errors)")
                        
                        # Find empty pass blocks in functions/classes (stubs)
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                                issues.append(f"[WARNING] Empty stub ({node.name}) found in {path} at line {node.lineno}")

                    # 3. Text Checks
                    if 'TODO' in content or 'FIXME' in content:
                        issues.append(f"[INFO] Unresolved TODO/FIXME found in {path}")
                        
                except Exception as e:
                    pass

            # Audit TSX/TS Files
            elif file.endswith(('.tsx', '.ts')):
                files_scanned += 1
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if 'TODO' in content or 'FIXME' in content:
                        issues.append(f"[INFO] Unresolved TODO/FIXME found in {path}")
                    if 'console.log(' in content:
                        issues.append(f"[WARNING] Leftover console.log in {path}")
                    if 'any' in content: # simplistic check
                        if re.search(r':\s*any\b', content):
                            issues.append(f"[WARNING] Usage of TypeScript 'any' type in {path}")
                except Exception:
                    pass
                    
            # Audit Infrastructure
            elif file.endswith(('.tf', '.yaml', '.yml')):
                files_scanned += 1
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if 'password' in content.lower() or 'secret' in content.lower():
                        # Basic heuristic
                        if not re.search(r'(env|secret_key_base|arn|parameter)', content.lower()):
                            issues.append(f"[WARNING] Possible hardcoded secret in {path}")
                except Exception:
                    pass

    print(f"Total files scanned: {files_scanned}")
    print("--- AUDIT RESULTS ---")
    for issue in issues:
        print(issue)
    
    if not issues:
        print("Repo is pristine!")

audit_repo()
