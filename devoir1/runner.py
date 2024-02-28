import subprocess

# for iter in range(1,6):
#     subprocess.run(["python3", "main.py", "--agent=advanced", "--group=easy", f"--number={iter}"])
#     subprocess.run(["mv", "visualization.png", f"visualizations/easy/solution_{iter}.png"])

for iter in range(1,11):
    subprocess.run(["python3", "main.py", "--agent=advanced", "--group=hard", f"--number={iter}"])
    # subprocess.run(["mv", "visualization.png", f"visualizations_choice/hard/solution_{iter}.png"])