import os
import glob

# .res 파일 삭제
res_files = glob.glob("*.res")
for file in res_files:
    os.remove(file)
print("All .res files have been deleted.")

sol_files = glob.glob("*.sol")
for file in res_files:
    os.remove(file)
print("All .sol files have been deleted.")