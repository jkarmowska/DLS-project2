import torch
from Bio import SeqIO
from evaluation_script import DeepSTARR_MultiTask, one_hot_encode 

def predict_ratio_only(model_path, sequence):
    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Load the model
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # 2. Encode the input sequence
    input_tensor = one_hot_encode(sequence).unsqueeze(0).to(device)

    # 3. Run the forward pass
    with torch.no_grad():
        # The model returns (logits_is_active, pred_rna_dna_ratio)
        # We only care about the second output
        _, pred_ratio = model(input_tensor)

    # 4. Extract the scalar value
    predicted_rna_dna_ratio = pred_ratio.item()

    return predicted_rna_dna_ratio

if __name__ == "__main__":
    path_to_model = "best_model.pth"
    seq_file = "subtaskA.fa"

    sample_seq = next(SeqIO.parse(seq_file, "fasta"))
    sample_dna_sequence = str(sample_seq.seq)  # Convert to string for prediction
    try:
        ratio = predict_ratio_only(path_to_model, sample_dna_sequence)

        print(f"DNA Sequence: {sample_dna_sequence}")
        print(f"Predicted RNA/DNA Ratio: {ratio:.4f}")

    except FileNotFoundError:
        print(f"Error: Could not find the model file at '{path_to_model}'.")