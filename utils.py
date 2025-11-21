import os
import torch
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
import imageio
import numpy as np

from pdb import set_trace as st

def save_checkpoint(epoch, model, args, best=False):
    if best:
        path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else:
        path = os.path.join(args.checkpoint_dir, 'model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer
        
def viz_cls(verts, pred_label, path, device, num_points=10000):
    """
    visualize classification result
    output: a 360-degree gif
    """
    import imageio
    import cv2
    import pytorch3d
    from pytorch3d.renderer import FoVPerspectiveCameras

    image_size = 256
    background_color = (1, 1, 1)
    N_frames = 60

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12 * i for i in range(N_frames)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(
        dist=dist, elev=elev, azim=azim, device=device
    )
    cameras = FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    # verts: (N, 3) on CPU → move to device, add batch dim
    sample_verts = verts.to(device=device, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)

    # Create dummy features (e.g., all ones = white points)
    # Shape should be (1, N, C); C=3 for RGB
    features = torch.ones_like(sample_verts)  # (1, N, 3)

    point_cloud = pytorch3d.structures.Pointclouds(
        points=sample_verts,
        features=features
    ).to(device).extend(N_frames)  # (N_frames, N, 3)

    renderer = get_points_renderer(
        image_size=image_size,
        background_color=background_color,
        device=device,
    )

    # Render: (N_frames, H, W, 3)
    rend = renderer(point_cloud, cameras=cameras).cpu().numpy()
    rend = (rend * 255).astype(np.uint8)

    # Add predicted label text to each frame
    pred_class = pred_label.item() if torch.is_tensor(pred_label) else pred_label
    map = {
        0: "chair",
        1: "vase",
        2: "lamp",
    }
    text = f"Predicted Class: {map[pred_class]}"
    for i in range(rend.shape[0]):
        cv2.putText(
            rend[i],
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

    imageio.mimsave(path, rend, fps=15)

def viz_seg (verts, labels, path, device, num_points=10000):
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)
    colors = [[1.0,1.0,1.0], [1.0,0.0,1.0], [0.0,1.0,1.0],[1.0,1.0,0.0],[0.0,0.0,1.0], [1.0,0.0,0.0]]

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.unsqueeze(0).repeat(30,1,1).to(torch.float)
    sample_labels = labels.unsqueeze(0)
    sample_colors = torch.zeros((1,num_points,3))

    # Colorize points based on segmentation labels
    for i in range(6):
        sample_colors[sample_labels==i] = torch.tensor(colors[i])

    sample_colors = sample_colors.repeat(30,1,1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)
    rend = (rend * 255).astype(np.uint8)

    imageio.mimsave(path, rend, fps=15)

