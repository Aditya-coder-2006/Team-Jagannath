import os

project_root = r"d:\BHU\code_S_H\iitbhu_S_H\project_root"

for root, _, files in os.walk(project_root):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            if '\\"\\"\\"' in content:
                print(f"Fixing {filepath}")
                content = content.replace('\\"\\"\\"', '"""')
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
print("Docstring patching complete!")
