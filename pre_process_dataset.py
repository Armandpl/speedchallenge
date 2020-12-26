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
    tmp_dir = "data/tmp"
    train_dir = "data/train"
    val_dir = "data/valid"
    test_dir = "data/test"
    val_pct = 0.3

    video_to_frames("./data/train.mp4", tmp_dir)
    # video_to_frames("./data/test.mp4", test_dir)

    os.makedirs(val_dir)
    os.makedirs(train_dir)
    nb_images = len([name for name in os.listdir(tmp_dir) if os.path.isfile(os.path.join(tmp_dir, name))])

    with open("data/train.txt") as f:
        annotations = f.readlines()

    train_annotations = open(os.path.join(train_dir, "annotations.txt"), "w")
    val_annotations = open(os.path.join(val_dir, "annotations.txt"), "w")

    train_count = 0
    val_count = 0
    for i in range(nb_images):
        if i in range(7140, 10199) or i in range(17340, 20399):
            val_annotations.write(annotations[i])
            fname = idx_to_fname(i)
            out_fname = idx_to_fname(val_count)
            shutil.move(os.path.join(tmp_dir, fname), os.path.join(val_dir, out_fname))
            val_count += 1
        else:
            train_annotations.write(annotations[i])
            fname = idx_to_fname(i)
            out_fname = idx_to_fname(train_count)
            shutil.move(os.path.join(tmp_dir, fname), os.path.join(train_dir, out_fname))
            train_count += 1

    train_annotations.close()
    val_annotations.close()
