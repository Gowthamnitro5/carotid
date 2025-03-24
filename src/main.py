import os
from model import carotidSegmentation, analyze_plaque, highlight_and_analyze_plaque
# from analysis import analyze_carotid_plaque, visualize_results
import cv2
from stack import load_image_slices,generate_3d_models
# from mmm import plot_pred
from visual import main as visualizer

def main():
    model = carotidSegmentation()
    
    # image_loc = 'data/Common Carotid Artery Ultrasound Images/US images/202202071357530054VAS_slice_557.png'
    image_loc = 'data/DATASET1/frame-04.jpg'
    
    pred_img_path = model.plot_prediction(image_loc,image_name='extracted')
    # pred_img_path = model.plot_pred(image_loc)
    analyze_plaque(pred_img_path)
    highlight_and_analyze_plaque(pred_img_path)

    # image_dir = 'data/Common Carotid Artery Ultrasound Images/US images/'
    # image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    # for idx, image_file in enumerate(image_files[:200]):  # Use slicing to select the first 5 images
    #     print(f"Processing image {idx + 1}: {image_file}...")

    #     image_loc = os.path.join(image_dir, image_file)

    #     # Generate prediction image
    #     pred_img_path = model.plot_prediction(image_loc, image_name={idx})
        
    #     # Analyze the plaque and highlight it
    #     analyze_plaque(pred_img_path)
    #     highlight_and_analyze_plaque(pred_img_path)

    #     # generate_3d_models(image_slices=load_image_slices('generated'),threshold=5)
    # visualizer('generated')


if __name__ == "__main__":
    main()