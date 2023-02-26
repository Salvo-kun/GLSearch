from models.gl_search import GLSearch
from options import opt

if __name__ == '__main__':       
    model = GLSearch(opt, 1, { }, 10)
    print(model)