import os
import cv2


traversal_file="archive/chest_xray/chest_xray"
output_file="archive/chest_xray_resized"
img_width_height=512

def resize_img(img_path,save_path):
    # open img
    img = cv2.imread(img_path)
    h, w = img.shape[0], img.shape[1]
    print("Original h : " + str(h) + "px Original w : " + str(w) + "px")
    rate=img_width_height/h
    img_processing = cv2.resize(img, (0, 0), fx=rate, fy=rate, interpolation=cv2.INTER_NEAREST)
    # change file name to jpg if png
    if save_path[-3:] == "png":
        save_path=save_path.replace("png", "jpg")
    cv2.imwrite(save_path, img_processing)
    print("Save as : " + save_path)

    pass

def show_files(path, all_files):
    # traverse all flies and folder
    file_list = os.listdir(path)
    # verifier its flies or folder
    for file in file_list:
        # get path
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files)
        else:
            all_files.append(path+"/"+file)
            #print(cur_path)
    return all_files

if os.path.isdir(traversal_file):
    print("Check traversal file ok")
else:
    print("Traversal file error")

if os.path.isdir(output_file):
    print("Check water mask file ok")
else:
    print("Water mask file warning,auto create it")
    os.mkdir(output_file)

#first traverse the folder
contents = show_files(traversal_file, [])
# then traverse treat each files
for content in contents:
    # print(content)
    # 判断是否为图片
    if content.endswith('jpeg') or content.endswith('png'):
        print("processing : "+content)
        resize_img(content,output_file + "/" +content[30:])

