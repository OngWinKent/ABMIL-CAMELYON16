from model import Attention, GatedAttention
import torch.nn as nn
import torch
from typing import *
import torch.utils.data as data_utils
from dataloader import Camelyon16Bags
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os

"""Retrieve attention model for training and inference"""
def init_model(model_name: str, in_features: int= 1024, patch_emb_size: int= 500, attn_hid_size: int= 128) -> nn.Module:
    if model_name not in['attention', 'gated_attention']:
        raise Exception("Select only attention or gated attention as model")
    print(f"[Model] model {model_name} selected")
    # Attention model
    if model_name ==  'attention':
        model = Attention(in_features= in_features, M= patch_emb_size, L=attn_hid_size)
    # Gated attention model
    else:
        model = GatedAttention(in_features= in_features, M= patch_emb_size, L=attn_hid_size)
    return model

"""Initialize device to be running on"""
def init_device(seed: int= 1) -> bool:
    is_cuda = True if torch.cuda.is_available() else False

    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.manual_seed(seed)
        print('[Device] Running on GPU')
    else:
        print('[Device] Running on CPU')

    return is_cuda

"""Load dataset"""
def load_dataset(root: str) -> Tuple[torch.utils.data.DataLoader,torch.utils.data.DataLoader]:
    print(f'[Dataset] Loading train and test dataset')
    # Train dataset
    train_loader = data_utils.DataLoader(
            Camelyon16Bags(root= root, train=True, features="UNI"),
            batch_size=1,  # Must be 1 due to varying patch sizes (N) across slides
            shuffle=True,
        )
    # Test dataset
    test_loader = data_utils.DataLoader(
        Camelyon16Bags(root= root, train=False, features="UNI"),
        batch_size=1,
        shuffle=False,
    )
    return train_loader, test_loader

"""Model training"""
def train(model: nn.Module, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, is_cuda: bool, train_params: dict, weights_path: str):
    print(f"[Training] Start model training")
    # Create weights saving parent folder
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    # Get model training parameters
    epoch_num = train_params.get('epoch_num')
    lr = train_params.get('lr')
    weight_decay = train_params.get('weight_decay')

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr= lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    # Main training loop
    best_test_loss = 1
    for epoch in tqdm(range(1, epoch_num+1), desc= 'Model training'):
        model.train()
        train_loss = 0.
        train_error = 0.
        for data, label in train_loader:
            bag_label = label[0]

            if is_cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)

            # reset gradients
            optimizer.zero_grad()
            # calculate loss and metrics
            loss, _ = model.calculate_objective(data, bag_label)
            train_loss += loss.data[0]
            error, _ = model.calculate_classification_error(data, bag_label)
            train_error += error
            # backward pass
            loss.backward()
            # step
            optimizer.step()

        # Calculate train loss and error for epoch
        avg_train_loss = train_loss / len(train_loader)
        avg_train_loss = round(avg_train_loss.cpu().numpy()[0], 4)
        avg_train_error = round(train_error/ len(train_loader), 4)

        # Compute test loss and error
        avg_test_loss, avg_test_error, avg_class_acc = test(model= model, test_loader= test_loader, is_cuda= is_cuda)

        # Save trained model with lowest test loss
        if avg_test_loss <= best_test_loss:
            print(f"best model saved at epoch: {epoch}, current test loss: {avg_test_loss} best test loss: {best_test_loss}")
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), weights_path)

        # Show training details 
        tqdm.write(f'Epoch: {epoch}, [Loss] Train|Test: {avg_train_loss:.4f}|{avg_test_loss}, [Error] Train|Test: {avg_train_error}|{avg_test_error}')
    print(f"[Training] Model finished training")

"""Test on test loader"""
def test(model: nn.Module, test_loader: torch.utils.data.DataLoader, is_cuda: bool) -> Tuple[float, float, float]:
    model.eval()
    test_loss = 0.0
    test_error = 0.0
    bag_correct_num = 0
    total_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            bag_label = label[0]

            if is_cuda:
                data, bag_label = data.cuda(), bag_label.cuda()

            # Forward Pass
            Y_prob, predicted_label, _ = model(data)
            loss, attention_weights = model.calculate_objective(data, bag_label)
            error, _ = model.calculate_classification_error(data, bag_label)
            # Compute loss
            test_loss += loss.item()
            test_error += error
            # Compute bag classification
            bag_gt = bag_label.cpu().numpy()[0]
            bag_pre = int(predicted_label.cpu().numpy()[0][0])
            if bag_gt == bag_pre:
                bag_correct_num += 1
            total_num += 1

    # Average 
    avg_test_loss = round(test_loss / total_num, 4)
    avg_test_error = round(test_error / total_num)
    avg_class_acc = round(bag_correct_num / total_num, 4)

    return avg_test_loss, avg_test_error, avg_class_acc

"""Running inference"""
def inference(model: nn.Module, test_loader: torch.utils.data.DataLoader, is_cuda: bool, show_plot: bool, weights_path: str):
    # Check input weights path existence
    if not os.path.exists(weights_path):
        raise Exception(f"Input weights path {weights_path} not found for inference")
    # Load model for inference
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    test_loss = 0.0
    test_error = 0.0
    bag_correct_num = 0
    total_num = 0

    with torch.no_grad():
        for data, label in tqdm(test_loader, desc= 'Inferencing'):
            bag_label = label[0]
            instance_labels = label[1]
            coords = label[2]

            if is_cuda:
                data, bag_label = data.cuda(), bag_label.cuda()

            # Forward Pass
            Y_prob, predicted_label, _ = model(data)
            loss, attention_weights = model.calculate_objective(data, bag_label)
            error, _ = model.calculate_classification_error(data, bag_label)

            test_loss += loss.item()
            test_error += error

            # Extract numpy arrays for plotting
            bag_gt = bag_label.cpu().numpy()[0]
            bag_pre = int(predicted_label.cpu().numpy()[0][0])
            prob_val = Y_prob.cpu().numpy()[0][0]
            tqdm.write(f"Bag GT: {bag_gt} Bag Pre: {bag_pre} Loss: {loss.item():.4f} Test Error: {error:.4f}")

            if bag_gt == bag_pre:
                bag_correct_num += 1
            total_num += 1

            # Convert tensors to NumPy array (squeezing batch dim=1)
            coords_np = coords.squeeze(0).cpu().numpy() if isinstance(coords, torch.Tensor) else np.array(coords)
            y_inst_np = instance_labels.squeeze(0).cpu().numpy() if isinstance(instance_labels, torch.Tensor) else np.array(instance_labels)
            a_np = attention_weights.detach().cpu().squeeze().numpy()

            # Plot attention maps for positive ground-truth slides (or all test slides)
            if show_plot:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Ground-Truth Patch Annotations
                normal_mask = y_inst_np == 0
                tumor_mask = y_inst_np == 1

                ax1.scatter(
                    coords_np[normal_mask, 0],
                    coords_np[normal_mask, 1],
                    c="lightslategrey",
                    s=2,
                    alpha=0.3,
                    label="Normal Patch",
                )

                if np.any(tumor_mask):
                    ax1.scatter(
                        coords_np[tumor_mask, 0],
                        coords_np[tumor_mask, 1],
                        c="crimson",
                        s=4,
                        alpha=0.8,
                        label="Tumor Patch",
                    )

                ax1.invert_yaxis()  # Pathology grid origin (0,0) is top-left
                ax1.set_title(
                    f"Ground-Truth Patch Annotations\nTrue Status: {bag_gt} ({'Tumor' if bag_gt == 1 else 'Normal'})",
                    fontsize=11,
                    fontweight="bold",
                )
                ax1.set_xlabel("X Coordinate")
                ax1.set_ylabel("Y Coordinate")
                ax1.grid(True, linestyle="--", alpha=0.3)
                ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

                # Model Predicted Attention Weights Density Map 
                sc = ax2.scatter(
                    coords_np[:, 0],
                    coords_np[:, 1],
                    c=a_np,
                    cmap="magma",
                    s=3,
                    alpha=0.8,
                )
                ax2.invert_yaxis()
                ax2.set_title(
                    f"Predicted Attention Weights Map\nModel Prediction: {bag_pre} (Tumor Prob: {prob_val:.4f})",
                    fontsize=11,
                    fontweight="bold",
                )
                ax2.set_xlabel("X Coordinate")
                ax2.set_ylabel("Y Coordinate")
                ax2.grid(True, linestyle="--", alpha=0.3)

                cbar = plt.colorbar(sc, ax=ax2)
                cbar.set_label("Attention Weight Score", rotation=270, labelpad=15)

                plt.suptitle(
                    f"Camelyon16 ABMIL Evaluation — Test Slide",
                    fontsize=13,
                    fontweight="bold",
                )
                plt.tight_layout()
                plt.show()
                plt.close(fig)

        # Show bag classification accuracy
        bag_class_acc = round((bag_correct_num / total_num) * 100, 4)
        print(f"Bag classification accuracy: {bag_class_acc} %")