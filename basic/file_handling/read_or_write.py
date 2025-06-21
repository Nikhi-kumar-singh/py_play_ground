def write_file1(file,content):
 with open(file,"w") as f:
  f.write(content)

def write_file2(file,content):
 with open(file,"w") as f:
  f.writelines(content)


def append_in_file(file,content):
 with open(file,"a") as f:
  f.write(content)



def read_file1(file):
 with open(file,"r") as f:
  content=f.read()   
  print(content)



def read_file2(file):
 with open(file,"r") as f:
  content=f.readlines()
  for line in content:
   print(line.strip())




def main():
 file="example.txt"
 content1='''
 hello nikhil 
 how are you?
 how are you doing?
 whar are you doinag these days.
 '''
 content2='''
 hello everyone ,
 nikhil here
 i am feeling pleased to write code in here!!!
 '''
 write_file2(file,content1)
 append_in_file(file,content2)
 read_file2(file)

if __name__=="__main__":
 main()
