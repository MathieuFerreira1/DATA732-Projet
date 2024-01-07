from multiprocessing import Pool
from time import time

def do_thing(x):
    return str(x**1000)[:5]

def do_things(l_of_x):
    return [do_thing(x) for x in l_of_x]

def split_list(l, n):
    k, m = divmod(len(l), n)
    return [l[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def main():
    t = time()

    p = Pool(6)

    l = sum(list(p.imap_unordered(do_things, split_list(range(10_000), 6))), [])

    print(len(l), l[:10])

    print(int((time()-t)*1000), 'ms')

if __name__ == "__main__":
    main()