import os
import cv2

if __name__ == "__main__":
    input_dir = "data/test/"
    seq_len = 10

    img = cv2.imread(os.path.join(input_dir, "00000.jpg"))

    for fname in os.listdir(input_dir):
        idx = int(fname.replace(".jpg", ""))
        idx = idx+seq_len-1
        new_fname = str(idx).zfill(5)+".jpg"
        new_fname = os.path.join(input_dir, new_fname)
        old = os.path.join(input_dir, fname) 
        
        os.rename(old, new_fname)

    
    for i in range(seq_len-1):
        new_fname = str(idx).zfill(5)+".jpg"
        new_fname = os.path.join(input_dir, new_fname)
        imwrite(new_fname, img)
