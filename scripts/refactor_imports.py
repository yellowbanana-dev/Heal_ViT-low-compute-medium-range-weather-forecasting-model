import os
import sys
import re

dirs_to_check = ['Training', 'ACC_RMSE_Eval', 'Power_Spectra', 'Core', 'Data_Config']
base_dir = '/storage/bharat/Heal_ViT_Refactored_feat_48swin'

import_map = {
    'config': 'Data_Config',
    'data': 'Data_Config',
    'model': 'Core',
    'graph': 'Core',
    'metrics': 'Core',
    'utils': 'Core',
    'acc': 'ACC_RMSE_Eval'
}

sys_path_code = """import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""

for d in dirs_to_check:
    dir_path = os.path.join(base_dir, d)
    if not os.path.isdir(dir_path):
        continue
    for f in os.listdir(dir_path):
        if f.endswith('.py') and f != '__init__.py':
            filepath = os.path.join(dir_path, f)
            with open(filepath, 'r') as file:
                content = file.read()
            
            lines = content.split('\n')
            new_lines = []
            imports_modified = False
            for line in lines:
                new_line = line
                # check imports
                match = re.search(r'^(from|import)\s+(config|data|model|graph|metrics|utils|acc)(\b.*)', line)
                if match:
                    imports_modified = True
                    stmt, mod, rest = match.groups()
                    if stmt == 'from':
                        new_line = f"from {import_map[mod]}.{mod}{rest}"
                    elif stmt == 'import':
                        # not expected to be used as 'import config' often, but just in case
                        new_line = f"from {import_map[mod]} import {mod}"
                new_lines.append(new_line)
            
            # If the file is a main script in a subfolder and not a core library (though Core libs could benefit too),
            # we should add sys.path.append at the top if it imports from our modules.
            if imports_modified:
                new_content = '\n'.join(new_lines)
                if 'sys.path.append' not in new_content:
                    # find the first import or from statement to insert before it
                    insert_idx = 0
                    for i, line in enumerate(new_lines):
                        if line.startswith('import ') or line.startswith('from '):
                            insert_idx = i
                            break
                    new_lines.insert(insert_idx, sys_path_code)
            
            with open(filepath, 'w') as file:
                file.write('\n'.join(new_lines))
            print(f"Updated {filepath}")
