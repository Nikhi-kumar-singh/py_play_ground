import math
print(math.sqrt(10))


import random
print(random.randint(1,10))
print(random.choice(["apple","banana","guava"]))


import os
print(os.getcwd())


import shutil
#shutil.copy("std_lib.txt","t001.txt")


import json
data={"name":"nikhil","age":23}
print(data)
print(type(data))

data=json.dumps(data)
print(data)
print(type(data))


data=json.loads(data)
print(data)
print(type(data))


import csv

print("writing the file:\n")
with open("example.csv",mode="w",newline="") as file:
 writer=csv.writer(file)
 writer.writerow(["name","age"])
 writer.writerow(["nikhil",23])
 writer.writerow(["manish",23])
 writer.writerow(["sahil",23])
 writer.writerow(["rahul",20])

print("reading the file:")
with open("example.csv",mode="r") as file:
 reader=csv.reader(file)
 for row in reader:
  print(row)



from datetime import datetime,timedelta
now=datetime.now()
print(now)
yesterday=now-timedelta(days=1)
print(now-yesterday)


import time
start=time.time()
time.sleep(2)
end=time.time()
print("time taken during sleep : ",end-start)


import re
pattern1 = r"\d+"
pattern2 = r"\d-"
text = "there are 123 apples 321"

match1 = re.search(pattern1, text)
match2 = re.search(pattern2, text)

if match1:
    print("matching elements : ", match1.group())
else:
    print("No match for pattern1")

if match2:
    print("matching elements : ", match2.group())
else:
    print("No match for pattern2")





