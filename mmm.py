import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure
from datetime import datetime

def plot_pred(self, image_loc, labels=False, text=True, image_name=None):
    image = self.get_image(image_loc)
    preds = self.predict(image_loc)
    pred_out = preds[0][0].detach().numpy()  # Artery prediction mask
    background = image[0][2].detach().numpy()
    
    # Plot background image
    plt.imshow(background, cmap='Greys_r', alpha=1)
    plt.imshow(pred_out, cmap='YlOrRd', alpha=pred_out * 0.5)
    
    # Add True Label (if provided)
    if labels:
        label_out = self.get_label(image_loc)[0][0]
        plt.imshow(label_out, cmap='RdYlBu', alpha=label_out * 0.5)
        if text:
            dice_loss = round(self.eval(image_loc), 4)
            plt.xlabel(f'Prediction = Red, True Label = Blue \n Dice Loss: {dice_loss}')
    else:
        if text:
            plt.xlabel('Prediction = Red')
    
    if text:
        plt.title('Carotid Artery Segmentation')
    
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    
    # Extract artery boundaries for inner and outer walls
    outer_wall = filters.sobel(pred_out) > 0  # Edge detection on artery mask
    
    # Inner wall extraction by erosion
    inner_wall_mask = morphology.binary_erosion(pred_out, morphology.disk(5))
    inner_wall = filters.sobel(inner_wall_mask) > 0
    
    # Residual lumen extraction (further erosion)
    lumen_area = morphology.binary_erosion(pred_out, morphology.disk(15))
    
    # Plot outer wall in red
    outer_wall_contours = measure.find_contours(outer_wall, level=0.5)
    for contour in outer_wall_contours:
        plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1, label="Outer Wall")

    # Plot inner wall in blue
    inner_wall_contours = measure.find_contours(inner_wall, level=0.5)
    for contour in inner_wall_contours:
        plt.plot(contour[:, 1], contour[:, 0], color='blue', linewidth=1, label="Inner Wall")
    
    # Overlay residual lumen area in light blue
    plt.imshow(lumen_area, cmap='Blues', alpha=0.3)
    
    # Add plaque thickness measurement in purple
    center_x, center_y = np.array(pred_out.shape) // 2
    thickness_x = [center_y, center_y + 10]
    thickness_y = [center_x, center_x + 10]
    plt.plot(thickness_x, thickness_y, color='purple', linewidth=1, label="Plaque Thickness")
    
    # Add legend
    plt.legend(['Outer Wall', 'Inner Wall', 'Residual Lumen', 'Plaque Thickness'])

    # Save or display the image
    if not image_name:
        dt = datetime.now().strftime(f"%d-%m-%Y")
        save_path = f'{dt}.png'
    else:
        save_path = f'{image_name}.png'
    
    plt.savefig(f'{save_path}.png', bbox_inches='tight')
    return f'{save_path}.png'
