from python import Python

#let np = Python.import_module("numpy")

def main():
    let torch = Python.import_module("torch")
    
    #print('torch', torch)


    print("Still seems to be python")
    
    # test range loop
    for x in range(0, 100) :
        print('x',x)
    
    #a = torch.tensor([1,2,3])
    #print('a', a)
    # list comprehension does not yet workk
    #value = [v for v in range(0, 100)]
    #print('v')