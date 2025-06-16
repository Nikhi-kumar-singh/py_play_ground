import multiprocessing as mlp
import time



def hello(n=10):
  if n<=0:
    return

  time.sleep(2)
  print(f"hello : {n}")
  hello(n-1)



def world(n=10):
  if n<=0:
    return

  time.sleep(2)
  print(f"world : {n}")
  world(n-1)



if __name__=="__main__":
  p1=mlp.Process(target=hello,args=(10,))
  p2=mlp.Process(target=world,args=(10,))

  start=time.time()

  p1.start()
  p2.start()

  p1.join()
  p2.join()

  end=time.time()

  print(f"time taken : {end-start}")
