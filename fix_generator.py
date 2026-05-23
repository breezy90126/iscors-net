
import os

file_path = 'datasets/xyt_dataset_generator.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
indent_mode = False
for i, line in enumerate(lines):
    if i == 2256: # Line 2257 (0-indexed 2256) is "# In[4]:"
        new_lines.append("\nif __name__ == '__main__':\n")
        indent_mode = True
    
    if indent_mode:
        new_lines.append("    " + line)
    else:
        new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Fixed xyt_dataset_generator.py")
