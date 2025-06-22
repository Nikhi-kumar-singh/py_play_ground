def fun1():
 try:
  a=b
 except:
  print("user defined error")
  print("variable is not assigned\n")



def fun2():
 try:
  a=b
 except NameError as ex:
  print(ex)


def fun3():
 try:
  a=10/0
 except NameError as ex:
  print(ex)
 except Exception as ex:
  print(ex)



def fun4():
 try:
  b=10
  a=int(input("enter a : "))
  result=b/a
 except ZeroDivisionError as ex:
  print(ex)
  a=int(input("enter a: "))
  result=b/a
  print(result)
 except Exception as ex:
  print(ex)
 else:         #this code will always be executed in any case
  print(f"result is : {result}")




def fun5():
 try:
  b=10
  a=int(input("enter a : "))
  result=b/a
 except ZeroDivisionError as ex:
  print(ex)
  result=fun5()
  print(result)
  return result
 except Exception as ex:
  print(ex)
 else:#this code will be executed in case of no exceptio
  print(f"result is : {result}")
 finally:#this code will always be executed in any case
  print(f"code is successfully executed")




def main():
 fun5()


if __name__=="__main__":
 main()
