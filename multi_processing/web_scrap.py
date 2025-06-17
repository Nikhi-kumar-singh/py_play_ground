'''
real-world example : multithreading I/O bound tasks 
scenerio : web scraping 
'''



import threading as th
import requests
import time 
from bs4 import BeautifulSoup as bs




def print_content(soup):
  print(f"fetched {len(soup.text)} characters from  {url}")


def fetch_content(url):
  response=requests.get(url)
  soup=bs(response.content,"html.parser")

  print_content(soup)





if __name__=="__main__":
  urls=[

   "http://cppreference.com/",
   "https://docs.python.org/3/https://docs.python.org/3/"

  ]
  
  threads=[]

  for url in urls:
    thread=th.Thread(target=fetch_content,args=(url,))
    threads.append(thread)
    thread.start()


  for thread in threads:
    thread.join()
