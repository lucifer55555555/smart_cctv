import torch

def test_model():
    model = torch.jit.load("models/violence_cnn_lstm.pt", map_location="cpu")
    model.eval()
    
    with torch.no_grad():
        zeros = torch.zeros(1, 30, 3, 224, 224)
        prob_zeros = torch.sigmoid(model(zeros)).item()
        
        ones = torch.ones(1, 30, 3, 224, 224)
        prob_ones = torch.sigmoid(model(ones)).item()
        
        rand1 = torch.rand(1, 30, 3, 224, 224)
        prob_rand1 = torch.sigmoid(model(rand1)).item()
        
        rand2 = torch.rand(1, 30, 3, 224, 224) * 10
        prob_rand2 = torch.sigmoid(model(rand2)).item()
        
    print(f"Zeros prob: {prob_zeros:.4f}")
    print(f"Ones prob:  {prob_ones:.4f}")
    print(f"Rand1 prob: {prob_rand1:.4f}")
    print(f"Rand2 prob: {prob_rand2:.4f}")

if __name__ == "__main__":
    test_model()
