import subprocess

# Filenames
file2 = 'DataACM.py'
file3 = '3025term_tf_idf.py'

# Function to run a Python file and check if it was successful
def run_script(file_path):
    try:
        result = subprocess.run(['python3', file_path], check=True)
        return result.returncode == 0  # True if the script was successful
    except subprocess.CalledProcessError:
        return False  # False if the script failed

# Run the first script
if run_script(file2):
    # If the first script was successful, run the second script
    if not run_script(file3):
        print(f"Execution failed at: {file3}")
else:
    print(f"Execution failed at: {file2}")
