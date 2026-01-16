import torch
from v2 import BigramLanguageModel, encode, decode

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = BigramLanguageModel()
model.to(device)

model.load_state_dict(torch.load('models/model_weights.pth', map_location=device))
model.eval()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=1500)[0].tolist()))
