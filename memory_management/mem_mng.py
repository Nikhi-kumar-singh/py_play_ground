import sys



'''
memory management : memory gets deallocated when it is referenced by none

reference count : the count of the references pointing to it
'''



a=[]
#it will print 2
# 1 point is referenced by a
# another is referenceed by the getrefcount function
print(sys.getrefcount(a))


b=a
# 3 will be printed as b is also referencing to a
print(sys.getrefcount(a))


del b

print(sys.getrefcount(a))




'''
garbage collection
constructor
destructor
'''


import gc


gc.enable()

gc.disable()


b=gc.collect()

print(f"number of garbage collected : {b}")

print(f"garbage collection stats : {gc.get_stats()}")


[]
print(f"unreachable garbage : {gc.garbage}")

