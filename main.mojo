from python import Python

def main():
    torch = Python.import_module("torch")
    
    print('torch', torch)
    
    a = torch.tensor([1,2,3])
    print('torch tensor', a)