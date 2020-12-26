import os
import cv2

def idx_to_fname(idx):
    return str(idx).zfill(5)+".jpg"

if __name__ == "__main__":
    input_dir = "data/test/"
    seq_len = 10

    img = cv2.imread(os.path.join(input_dir, "00000.jpg"))

    for i in reversed(range(10797+1)):
        idx = i+seq_len-1
        new_fname = idx_to_fname(idx)
        new_fname = os.path.join(input_dir, new_fname)
        old = os.path.join(input_dir, idx_to_fname(i)) 
        print(idx)
        print(old)
        print(new_fname)
        
        os.rename(old, new_fname)

    for i in range(seq_len-1):
        new_fname = str(i).zfill(5)+".jpg"
        new_fname = os.path.join(input_dir, new_fname)
        cv2.imwrite(new_fname, img)
