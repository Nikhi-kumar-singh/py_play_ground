def add(a,b):
    print(1)
    return a+b


def add(a=0,b=0,c=0):
    print(2)
    return a+b+c


def add(a=0,b=0,c=0,d=0):
    print(3)
    return a+b+c+d


if __name__ == "__main__":
    ans = add(10,20)
    print(ans)
