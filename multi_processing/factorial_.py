import multiprocessing  as mlp
import time
import sys
import math




sys.set_int_max_str_digits(100000)


def fact(n):
  if n<2:
    return n
  #print(f"start computing factorial of {n}")
  ans=fact(n-1)*n
  #print(f"factorial for {n} : {ans}")
  return ans




if __name__=="__main__":
  numbers=[500,200,300,400]
  start_time=time.time()
  
  with mlp.Pool() as pool : 
    results=pool.map(fact,numbers)

  print(f"results are : {results}")

  end_time=time.time()

  print(f"time taken : {end_time-start_time}")
