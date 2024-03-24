import subprocess

# for iter in ['A', 'B', 'C']:
#     subprocess.run(["python3", "main.py", "--agent=d3", f"--infile=./instances/instance{iter}.txt", "--no-viz"])
    
for iter in ['D', 'E']:
    subprocess.run(["python3", "main.py", "--agent=d3", f"--infile=./instances/instance{iter}.txt", "--no-viz", "--time=600"])