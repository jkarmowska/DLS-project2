import sys
import torch
import pandas as pd
# Import the model architecture and encoding function from model.py file
from project2.model import DNAModel, dna_one_hot 


def evaluate(model_path, data_path):
    """Loads a trained model and generates predictions for a given test dataset.
       Outputs results to stdout in a tab-separated format.
    """
    # 1. Model Loading
    # Initialize the architecture and load the saved weights
    model = DNAModel()
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set to evaluation mode

    # 2. Data Loading
    # Load the TSV file. The script expects 'sequence' and an ID column.
    df = pd.read_csv(data_path, sep='\t')
    
    # Identify the ID column dynamically
    id_col = df.columns[0]
    
    # 3. Prediction Generation
    # Print the required header for the output TSV
    print("id\tpredicted_is_active\tpredicted_rna_dna_ratio")
    
    # Disable gradient calculation for faster inference and lower memory usage
    with torch.no_grad():
        for _, row in df.iterrows():
            # Encode sequence and add batch dimension (1, 4, L)
            seq_tensor = torch.tensor(dna_one_hot(row['sequence'])).unsqueeze(0)
            
            # Forward pass through the model
            pred_cls, pred_reg = model(seq_tensor)
            
            # Convert classification probability to binary label (threshold 0.5)
            is_active = 1 if pred_cls.item() > 0.5 else 0
            
            # Get continuous regression value
            ratio = pred_reg.item()
            
            # Print the result for the current sequence
            print(f"{row[id_col]}\t{is_active}\t{ratio}")

if __name__ == "__main__":
    # Check if correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python evaluation_script.py <model_path.pth> <test_data.tsv>")
        sys.exit(1)

    # Execute evaluation using command line arguments
    # sys.argv[1] = path to .pth model
    # sys.argv[2] = path to .tsv data
    evaluate(sys.argv[1], sys.argv[2])