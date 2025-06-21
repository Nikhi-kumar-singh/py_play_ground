
def write_file(file,content):
 with open(file,'wb') as f:
  f.write(content)


def read_file1(file):
 with open(file,"rb") as f:
  content=f.read()
  print(content)


def read_file2(file):
 with open(file,"rb") as f:
  content=f.readlines()
  for line in content:
   print(line)


def main():
 file="example.bin"
 content=b"\x00\x01\x02\x03\x04"
 write_file(file,content)
 read_file1(file)
 read_file2(file)

if __name__=="__main__":
 main()
