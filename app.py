import tkinter as tk
from tkinter import filedialog
import cv2
from ultralytics import YOLO
import threading
import os
import time
import torch

def detect_and_display(source_path):
    """
    Runs YOLO detection on the source (image or video) and displays results in a popup.
    """
    # Check for GPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = YOLO('best.pt')
    
    # Custom label override
    if 1 in model.names and model.names[1] == 'Car':
        model.names[1] = 'Auto'
    
    ext = os.path.splitext(source_path)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv']
    
    if is_video:
        cap = cv2.VideoCapture(source_path)
        
        counted_ids = set()
        total_count = 0
        
        # For FPS calculation
        prev_time = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # --- Performance Optimization: Resize Frame ---
            # Processing high-res video is slow. Resizing to 640 width significantly speeds it up.
            target_width = 640
            h, w, _ = frame.shape
            ratio = target_width / w
            target_height = int(h * ratio)
            frame_resized = cv2.resize(frame, (target_width, target_height))
            


            # Run inference with tracking and tuned thresholds
            results = model.track(
                frame_resized, 
                persist=True, 
                conf=0.3, 
                iou=0.45, 
                device=device,
                verbose=False
            )
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                for track_id in track_ids:
                    if track_id not in counted_ids:
                        total_count += 1
                        counted_ids.add(track_id)
            
            # Visualize results on the resized frame for speed
            annotated_frame = results[0].plot(labels=False)
            
            # Display Count
            cv2.putText(annotated_frame, f"Total Unique Vehicles: {total_count}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # --- Performance Tracking: FPS ---
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow("Vehicle Detection - Press 'q' to Exit", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
    else:
        # It's an image
        results = model(source_path, device=device)
        total_vehicles = len(results[0].boxes)
        annotated_img = results[0].plot(labels=False)
        cv2.putText(annotated_img, f"Total Vehicles: {total_vehicles}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow("Vehicle Detection", annotated_img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

def start_detection():
    file_path = filedialog.askopenfilename(
        title="Select Image or Video",
        filetypes=[
            ("Media Files", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp"),
            ("All Files", "*.*")
        ]
    )
    if file_path:
        detect_and_display(file_path)

def main():
    root = tk.Tk()
    root.title("Vehicle Detection Launcher")
    root.geometry("300x150")
    
    tk.Label(root, text="Select an Image or Video\nto detect vehicles", pady=20).pack()
    tk.Button(root, text="Select File", command=start_detection, padx=20, pady=10, bg="#4CAF50", fg="white").pack()
    
    root.mainloop()

if __name__ == "__main__":
    main()
