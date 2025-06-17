'''
this code is needs to be fixed it is not working
'''


import tracemalloc


def create_list():
 return [i for i in range(100000)]


def main():
 tracemalloc.start()

 create_list()

 snapshot=tracemalloc.take_snapshot()
 top_stats=snapshot.statistics("lineno")

 print("[ top 10]")
 for stat in top_stats[:10]:
  print(stat)



if __name__=="__main__":
 main()
