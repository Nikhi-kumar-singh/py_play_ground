'''
this is an advanced way of doing this
'''

from concurrent.futures import ThreadPoolExecutor as th
import time 


def print_number(n):
  time.sleep(1)
  return f"number : {n}"



def thread_runner(values): 
  with th(max_workers=3) as executor:
    res=executor.map(print_number,values)

  return res

def print_list(res):
  for x in res:
    print(x)




if __name__ == "__main__":
  start=time.time()
  n=10  
  res=thread_runner(range(10))
  print_list(res)
  end=time.time()

  print(f"time taken : {end-start}")
