
import os

image_files = []
os.chdir(os.path.join("data_test_rohit", "own_data_dir","waste_container"))
for filename in os.listdir(os.getcwd()):
    print("-gen--train-->>--filename----",filename)
    if filename.endswith(".jpg"):
        image_files.append("data_test_rohit/own_data_dir/waste_container" + filename)
#os.chdir("..")
with open("train_waste_container.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")
