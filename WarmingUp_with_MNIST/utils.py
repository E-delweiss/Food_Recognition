import torch

def device()->torch.device:
    """
    Set the device to 'cpu', 'cuda' or 'mps'.

    Args:
        None.

    Return:
        device : torch.device
    """

    ### Choosing device between CPU or GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # elif torch.has_mps:
    #     device=torch.device('mps')
    else:
        device=torch.device('cpu')
    
    print("\n------------------------------------")
    print(f"Execute notebook on - {device} -")
    print("------------------------------------\n")

    return device



if __name__ == "__main__":
    device()