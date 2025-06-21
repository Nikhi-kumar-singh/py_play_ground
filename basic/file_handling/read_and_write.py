def read_write(file):
 with open(file,"w+") as f:
  f.write("hello world\n")
  f.write("nikhil here\n")

  #move the cursor to the 0 index
  f.seek(0)
  content=f.read()
  print(content)



def main():
 file="example1.txt"
 read_write(file)


if __name__=="__main__":
 main()
