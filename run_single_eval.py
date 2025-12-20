import subprocess
import sys

result = subprocess.run([sys.executable, "main.py"], capture_output=False, text=True)
#print("평가 완
!")
