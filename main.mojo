from python import Python

def main():
    let torch = Python.import_module("torch")
    
    print('torch', torch)
    
    let a = torch.tensor([1,2,3])
    print('torch tensor', a)