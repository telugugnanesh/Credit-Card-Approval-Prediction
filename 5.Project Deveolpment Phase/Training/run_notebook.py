import json
import io
import sys
from contextlib import redirect_stdout
def main():
    print("Executing Jupyter Notebook: loan_approval_ml.ipynb...")
    try:
        with open('loan_approval_ml.ipynb', 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except FileNotFoundError:
        print("Error: loan_approval_ml.ipynb not found.")
        sys.exit(1)
    global_env = {'__name__': '__main__'}
    exec_count = 1
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            code = "".join(cell['source'])
            print(f"Executing Code Cell {exec_count}...")
            
            f_stdout = io.StringIO()
            try:
                with redirect_stdout(f_stdout):
                    exec(code, global_env)
                output_text = f_stdout.getvalue()
                
                # Update cell outputs in notebook
                cell['outputs'] = []
                if output_text:
                    cell['outputs'].append({
                        "name": "stdout",
                        "output_type": "stream",
                        "text": output_text.splitlines(keepends=True)
                    })
                cell['execution_count'] = exec_count
            except Exception as e:
                print(f"Error executing cell {exec_count}: {e}")
                import traceback
                tb_lines = traceback.format_exception(*sys.exc_info())
                cell['outputs'] = [{
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "output_type": "error",
                    "traceback": tb_lines
                }]
                cell['execution_count'] = exec_count
                
                # Write back the failed state to see the error in IDE
                with open('loan_approval_ml.ipynb', 'w', encoding='utf-8') as f:
                    json.dump(nb, f, indent=1)
                sys.exit(1)
            
            exec_count += 1
    # Save the executed notebook back
    with open('loan_approval_ml.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Jupyter Notebook executed successfully and saved with outputs!")
if __name__ == "__main__":
    main()
