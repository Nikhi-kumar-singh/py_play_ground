


def generate_num(n):
 for i in range(n):
  yield i


if __name__=="__main__":
 n=1000
 for num in generate_num(n):
  print(num)
  if num >= 10 : break
