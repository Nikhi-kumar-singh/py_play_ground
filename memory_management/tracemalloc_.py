'''
this code is needs to be fixed it is not working
'''


import tracemalloc


def create_list(n):
 return [i for i in range(n)]


def main():
 tracemalloc.start()
 n=1000000
 l1=create_list(n)

 snapshot=tracemalloc.take_snapshot()
 top_stats=snapshot.statistics("lineno")

 print("[ top 10]")
 for stat in top_stats[:10]:
  print(stat)



if __name__=="__main__":
 main()
