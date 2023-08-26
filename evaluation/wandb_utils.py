import wandb
import numpy as np
import monai
import torch
from PIL import Image
import matplotlib.pyplot as plt

warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="zeros")
 
def log_model_wandb(model_path):
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)
    wandb.run.log_artifact(artifact)

def create_wandb_table( columns):
    wandb_table = wandb.Table(columns=columns)
    return wandb_table
    
  
def log_wandb_table_img(pair_img_results, wandb_table, table_name):
    # take list of dictionaries and convert to wandb table
    for pair in pair_img_results:
        wandb_table.add_data(pair["fix_name"], pair["mov_name"], wandb.Image(Image.open(pair["fix_img"])), wandb.Image(Image.open(pair["mov_img"])), 
                             wandb.Image(Image.open(pair["warp_img"])), wandb.Image(Image.open(pair["flow"])),pair["ncc"])
        
    wandb.log({table_name: wandb_table})
        
def log_wandb_table_mask(pair_img_results, wandb_table, table_name):
    # take list of dictionaries and convert to wandb table
    for pair in pair_img_results:
        wandb_table.add_data(pair["fix_name"], pair["mov_name"], wandb.Image(Image.open(pair["fix_mask"])), wandb.Image(Image.open(pair["mov_mask"])), 
                             wandb.Image(Image.open(pair["warp_mask"])), pair["dice"], pair["hd95"])
    wandb.log({table_name: wandb_table})
        
def log_wandb_table_metrics(pair_img_results, wandb_table, table_name):
    # take list of dictionaries and convert to wandb table
    for pair in pair_img_results:
        wandb_table.add_data(pair["fix_name"], pair["mov_name"], pair["dice"], pair["hd95"], pair["tre"], pair["ncc"])
    wandb.log({table_name: wandb_table})
        
def log_wandb_table_kp(pair_img_results, wandb_table, table_name):
    # take list of dictionaries and convert to wandb table
    for pair in pair_img_results:
        wandb_table.add_data(pair["fix_name"], pair["mov_name"], wandb.Image(Image.open(pair["fix_kp"])), wandb.Image(Image.open(pair["mov_kp"])), 
                             wandb.Image(Image.open(pair["warp_kp"])), pair["tre"])
    wandb.log({table_name: wandb_table})
    
def plot_img(img, file_path, cmap=None):
    plt.imshow(img, cmap=cmap)
    plt.savefig(file_path)
    plt.close()

 
def plot_keypoints(fix_lms_warped,moved_img,slice_ , file_path):     
       
    kpts_fix_world=fix_lms_warped[abs(fix_lms_warped[:,1]-slice_)<2]
    plt.imshow(moved_img[:,slice_].T,'gray')
    plt.plot(kpts_fix_world[:,0], kpts_fix_world[:,2], 'r+')
    plt.colorbar()    
    plt.savefig(file_path) 
    plt.close()         
    
def plot_2D_deformation(vector_field, grid_spacing, **kwargs):
    """
    Interpret vector_field as a displacement vector field defining a deformation,
    and plot an x-y grid warped by this deformation.

    vector_field should be a tensor of shape (2,H,W)
    """
    _, H, W = vector_field.shape
    grid_img = np.zeros((H, W))
    grid_img[np.arange(0, H, grid_spacing), :] = 1
    grid_img[:, np.arange(0, W, grid_spacing)] = 1
    grid_img = torch.tensor(grid_img, dtype=vector_field.dtype).unsqueeze(0)  # adds channel dimension, now (C,H,W)
   
    grid_img_warped = warp(grid_img.unsqueeze(0), vector_field.unsqueeze(0))[0]
    plt.imshow(grid_img_warped[0], origin="lower", cmap="gist_gray")


def preview_3D_deformation(vector_field, grid_spacing, **kwargs):
    """
    Interpret vector_field as a displacement vector field defining a deformation,
    and plot warped grids along three orthogonal slices.

    vector_field should be a tensor of shape (3,H,W,D)
    kwargs are passed to matplotlib plotting

    Deformations are projected into the viewing plane, so you are only seeing
    their components in the viewing plane.
    """
    x, y, z = np.array(vector_field.shape[1:]) // 2  # half-way slices
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plot_2D_deformation(vector_field[[1, 2], x, :, :], grid_spacing, **kwargs)
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plot_2D_deformation(vector_field[[0, 2], :, y, :], grid_spacing, **kwargs)
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plot_2D_deformation(vector_field[[0, 1], :, :, z], grid_spacing, **kwargs)
    plt.savefig(kwargs["file_path"])
    plt.close()
    
def preview_image(image_array, normalize_by="volume", cmap=None, figsize=(12, 12), threshold=None, title="", file_path=None):
    """
    Display three orthogonal slices of the given 3D image.

    image_array is assumed to be of shape (H,W,D)

    If a number is provided for threshold, then pixels for which the value
    is below the threshold will be shown in red
    """
    if normalize_by == "slice":
        vmin = None
        vmax = None
    elif normalize_by == "volume":
        vmin = 0
        vmax = image_array.max().item()
    else:
        raise (ValueError(f"Invalid value '{normalize_by}' given for normalize_by"))

    # half-way slices
    x, y, z = np.array(image_array.shape) // 2
    imgs = (image_array[x, :, :], image_array[:, y, :], image_array[:, :, z])

    fig, axs = plt.subplots(1, 3, figsize=figsize)
#     fig.suptitle(title)
    for ax, im in zip(axs, imgs):
        ax.axis("off")
        ax.imshow(im, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)

        # threshold will be useful when displaying jacobian determinant images;
        # we will want to clearly see where the jacobian determinant is negative
        if threshold is not None:
            red = np.zeros(im.shape + (4,))  # RGBA array
            red[im <= threshold] = [1, 0, 0, 1]
            ax.imshow(red, origin="lower")
    
   
    plt.savefig(file_path)
    plt.close()

                                                                        