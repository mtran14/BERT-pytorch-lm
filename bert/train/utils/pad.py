import torch

def pad_masking(x):
    # x: (batch_size, seq_len)
    # padded_positions = x == PAD_INDEX
    # return padded_positions.unsqueeze(1)
    #output: (batch_size. 1, seq_len)
    #False/0  if not eq 0 else 1
    batch_size, seq_len = x.size()[0], x.size()[1]
    isC = len(x.size()) == 3

    device_id = x.get_device()
    device = 'cpu' if device_id < 0 else 'cuda:'+str(device_id)
    mask = torch.zeros(batch_size, 1, seq_len).to(device)

    for i in range(batch_size):
        for j in range(seq_len):
            if(not isC and x[i][j] == 0):
                mask[i][0][j] = 1
            elif(isC and x[i][j].sum() == 0):
                mask[i][0][j] = 1
    return mask > 0
