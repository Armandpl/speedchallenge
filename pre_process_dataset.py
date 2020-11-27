import os
import cv2
import shutil

def idx_to_fname(idx):
    return str(idx).zfill(5)+".jpg"

def video_to_frames(video_path, output_dir):
    os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)

    if cap.isOpened() == False:
        print("error opening : ", video_path)

    count = 0

    while(cap.isOpened()):

        ret, frame = cap.read()

        if ret == True:
            fpath = os.path.join(output_dir, idx_to_fname(count))
            cv2.imwrite(fpath, frame)
        else:
            break

        count +=1
    cap.release()

if __name__ == "__main__":
    train_dir = "data/train"
    val_dir = "data/valid"
    test_dir = "data/test"
    val_pct = 0.3

    video_to_frames("./data/train.mp4", train_dir)
    # video_to_frames("./data/test.mp4", test_dir)

    os.makedirs(val_dir)
    train_pct = 1 - val_pct
    nb_images = len([name for name in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, name))])
    val_idx = int(train_pct*nb_images)

    with open("data/train.txt") as f:
        annotations = f.readlines()

    train_annotations = open(os.path.join(train_dir, "annotations.txt"), "w")
    val_annotations = open(os.path.join(val_dir, "annotations.txt"), "w")

    j = 0
    for i in range(nb_images):
        if i >= val_idx:
            val_annotations.write(annotations[i])
            fname = idx_to_fname(i)
            out_fname = idx_to_fname(j)
            shutil.move(os.path.join(train_dir, fname), os.path.join(val_dir, out_fname))
            j += 1
        else:
            train_annotations.write(annotations[i])

    train_annotations.close()
    val_annotations.close()
