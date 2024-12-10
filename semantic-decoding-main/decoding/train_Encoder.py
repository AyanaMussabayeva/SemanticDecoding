import os
import time
import logging
import numpy as np
import json
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
torch.cuda.empty_cache()

from sklearn.model_selection import train_test_split

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim
from utils_resp import get_resp

# adding my neural networks 
from EncodingNN import EncodingNN
from RidgeRegressionNN import RidgeRegressionNN
import ipdb


np.random.seed(42)
torch.manual_seed(42)

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler("encoder_training_ml804.log"),  # Save logs to file
        logging.StreamHandler()  # Print logs to console
    ]
)
logger = logging.getLogger(__name__)  # Create logger





if __name__ == "__main__":
    logger.info(f"Execution start")
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
   



    parser = argparse.ArgumentParser()
    
    # This part of code is provided by the authors to get the semantic features from the stimuli using language model.
    # It takes lots of time, so I saved teh features in pkl file 
    
    # parser.add_argument("--subject", type=str, required=True)
    # parser.add_argument("--gpt", type=str, default="perceived")
    # parser.add_argument("--sessions", nargs="+", type=int, 
    #                     default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    # args = parser.parse_args()

    # Load training stories
    # stories = []
    # with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
    #     sess_to_story = json.load(f)
    # for sess in args.sessions:
    #     stories.extend(sess_to_story[str(sess)])

    # # Load GPT model
    # with open(os.path.join(config.DATA_LM_DIR, args.gpt, "vocab.json"), "r") as f:
    #     gpt_vocab = json.load(f)
    # gpt_model_path = os.path.join(config.DATA_LM_DIR, args.gpt, "model")
    # gpt_device = "cpu"
    # gpt = GPT(path=gpt_model_path, vocab=gpt_vocab, device=gpt_device)
    # features = LMFeatures(model=gpt, layer=config.GPT_LAYER, context_words=config.GPT_WORDS)
    #ipdb.set_trace()
    
    
    
    parser.add_argument("--pickle_path", type=str, required=True, help="semantic_data.pkl")
    parser.add_argument("--nn", type=str, default="RidgeRegressionNN")
    parser.add_argument("--opt",type=str, default="SGD" )
    args = parser.parse_args()

    # Load the saved data from the pickle file
    logger.info(f"Attempting to load data...")
    if os.path.exists(args.pickle_path):
        with open(args.pickle_path, "rb") as f:
            data = pickle.load(f)
        print(f"Data successfully loaded from {args.pickle_path}")
        logger.info(f"Data successfully loaded from {args.pickle_path}")
    else:
        logger.error(f"Pickle file not found at {args.pickle_path}")
        raise FileNotFoundError(f"Pickle file not found at {args.pickle_path}")
    
    #ipdb.set_trace()

    # Extract data
    rstim = data["rstim"]
    rresp = data["rresp"]
    
    #ipdb.set_trace()



    # # Generate stimulus features
    # rstim, tr_stats, word_stats = get_stim(stories, features)
    # # Load brain responses
    # rresp = get_resp(args.subject, stories, stack=True)

    # Train-test split
    rstim_train, rstim_val, rresp_train, rresp_val = train_test_split(
        rstim, rresp, test_size=0.2, random_state=42
    )
    
    
    
    # UPDATE: Create DataLoader for batch processing
    batch_size = 64  # You can adjust this based on your GPU/CPU capacity
    train_dataset = TensorDataset(torch.tensor(rstim_train, dtype=torch.float32), 
                                   torch.tensor(rresp_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.tensor(rstim_val, dtype=torch.float32), 
                                 torch.tensor(rresp_val, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    
    
    #ipdb.set_trace()
    # Define the neural network
    input_size = rstim.shape[1]
    output_size = rresp.shape[1]
    logger.info(f'Input size {input_size}')
    logger.info(f'Output size {output_size}')
    
    
    
    logger.info('Defining the network')
   
    
    if args.nn == 'EncodingNN':
        model = EncodingNN(input_size=input_size, output_size=output_size).to(device)
    elif args.nn == 'RidgeRegressionNN':
        alpha = 0.1
        model = RidgeRegressionNN(input_size=input_size, output_size=output_size, alpha=alpha).to(device)
    else:
        raise ValueError(f"Unknown neural network type: {args.nn}")


    criterion = nn.MSELoss()
    
    
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.opt == 'Adam': 
        optimizer  = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999)
    elif args.opt == 'RMSProp': 
        if args.nn == 'EncodingNN': 
            optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005, alpha=0.99)
        elif args.nn == 'RidgeRegressionNN':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
    else:
        raise ValueError(f"Unknown neural network type: {args.opt}")
    
 

    
    
    # Training the neural network
    epochs = 100
    total_samples_processed = 0  # UPDATE: For throughput
    prev_val_loss = None         # UPDATE: For statistical efficiency
    stat_efficiency = []         # UPDATE: To store statistical efficiency values


    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device) # Move inputs and targets to the GPU
            optimizer.zero_grad()
            preds, noise_structure = model(inputs, targets)
            pred_loss = criterion(preds, targets)
            noise_loss = torch.mean((targets - preds).var(dim=0))  # Variance as a proxy for noise
            total_loss = pred_loss + 0.1 * noise_loss
            total_loss.backward()
            optimizer.step()

            total_samples_processed += len(inputs)  # UPDATE: Increment total samples processed
           

            # Optionally log GPU stats per batch
            logger.info(
                f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
            )

        # End of epoch throughput calculation
        epoch_time = time.time() - epoch_start_time  # Time for epoch
        throughput = total_samples_processed / epoch_time  # Samples per second
        logger.info(f"Epoch [{epoch+1}/{epochs}], Throughput: {throughput:.2f} samples/second")
       

        #ipdb.set_trace()
        # Validation
        model.eval()
        with torch.no_grad():
            val_inputs = torch.tensor(rstim_val, dtype=torch.float32)
            val_targets = torch.tensor(rresp_val, dtype=torch.float32)
            val_preds, val_noise = model(val_inputs, val_targets)
            val_loss = criterion(val_preds, val_targets)
            logger.info(f"Validation Loss: {val_loss.item():.4f}")
            #ipdb.set_trace()
        
        if prev_val_loss is not None:
            improvement = prev_val_loss - val_loss
            samples_seen = total_samples_processed
            logger.info(f"Epoch [{epoch+1}/{epochs}]")

        prev_val_loss = val_loss
        logger.info(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}")
            
        #ipdb.set_trace()

    # Save the trained model and stats
    logger.info("Saving trained model and stats...")
    save_location = os.getcwd()
    os.makedirs(save_location, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_location, f"encoding_model_1.pth"))