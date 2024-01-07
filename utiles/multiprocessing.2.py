from multiprocessing import Pool
from time import time

def do_thing(x):
    return str(x**1000)[:5]

def main():
    t = time()

    p = Pool(6)

    l = list(p.imap_unordered(do_thing, range(10_000)))
    p.close()
    p.join()

    print(len(l), l[:10])

    print(int((time()-t)*1000), 'ms')

if __name__ == "__main__":
    main()