'''
this programs are for understanding about the circular referencing
'''


import gc 
import sys



class myobj:
  def __init__(self,name):
    self.name=name
    print(f"object created : {self.name}")

  def __del__(self):
    print(f"object deleted : {self.name}")  



if __name__=="__main__":
  obj1=myobj("nikhil")
  obj2=myobj("nikhilsingh")
  
  obj1.ref=obj2
  obj2.ref=obj1

  del obj1
  del obj2
 

  #print(f"object1 references : {sys.getrefcount(obj1)}")
  #print(f"object1 references : {sys.getrefcount(obj2)}")
 

  gc.collect()
  
  print(f"garbage status : {gc.garbage}")
