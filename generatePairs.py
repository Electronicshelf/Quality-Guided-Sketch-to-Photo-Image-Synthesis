import os
import random

class GeneratePairs:
    """
    Generate the pairs.txt file that is used for training face classifier when calling python `src/train_softmax.py`.
    Or others' python scripts that needs the file of pairs.txt.
    Doc Reference: http://vis-www.cs.umass.edu/lfw/README.txt
    """

    def __init__(self, data_dir, pairs_filepath, img_ext):
        """
        Parameter data_dir, is your data directory.
        Parameter pairs_filepath, where is the pairs.txt that belongs to.
        Parameter img_ext, is the image data extension for all of your image data.
        """
        self.data_dir = data_dir
        self.data_dir2 = data_dir
        self.pairs_filepath = pairs_filepath
        self.img_ext = img_ext


    def generate(self):
        self._generate_matches_pairs()
        self._generate_mismatches_pairs()



    def _generate_matches_pairs(self):
        """
        Generate all matches pairs
        """
        random.seed(random.randint(30, 90))
        count = 0
        data_dir = (os.listdir(self.data_dir))
        random.shuffle(data_dir)
        # self.data_dir = data_dir
        # # self.data_dir = data_dir
        for name in(data_dir):

            if name == ".DS_Store":
                continue
            # print(self.data_dir + name)
            # exit()
            a = []
            # print(os.listdir(self.data_dir))
            # data_dir  = (os.listdir(self.data_dir))
            # random.shuffle(data_dir)
            # # self.data_dir = data_dir
            # print(data_dir)
            # exit()
            for file in os.listdir(self.data_dir + name):
                if file == ".DS_Store":
                    continue
                a.append(file)

            count += 1
            with open(self.pairs_filepath, "a") as f:
                for i in range(1):
                    print(count)
                    if count <= 150:
                        w = name  # [0] + "_" + temp[1]
                        l = random.choice(a).split(".")[0]  # .lstrip("0").rstrip(self.img_ext)
                        r = random.choice(a).split(".")[0]  # .lstrip("0").rstrip(self.img_ext)
                        print(w, l, r)
                        # exit()
                        f.write(w + "\t" + l + "\t" + r + "\n")

                    # temp = random.choice(a).split(".")[0] # This line may vary depending on how your images are named.
                    # print(name)
                    # print(temp)
                    # exit()
                    # w = temp#[0] + "_" + temp[1]




    def _generate_mismatches_pairs(self):
        """
        Generate all mismatches pairs
        """
        random.seed(random.randint(91, 200))
        count = 0
        data_dir = (os.listdir(self.data_dir))
        random.shuffle(data_dir)
        for i, name in enumerate((data_dir)):
            if name == ".DS_Store":
                continue

            remaining = (data_dir)
            remaining = [f_n for f_n in remaining if f_n != ".DS_Store"]
            del remaining[i] # deletes the file from the list, so that it is not chosen again
            other_dir = random.choice(remaining)
            count += 1
            with open(self.pairs_filepath, "a") as f:
                for i in range(1):
                    if count <= 300:
                        file1 = random.choice((os.listdir(self.data_dir + name)))
                        file2 = random.choice((os.listdir(self.data_dir + other_dir)))
                        f.write(name + "\t" + file1.split(".")[0] + "\t" + other_dir + "\t" + file2.split(".")[0] + "\n")#.lstrip("0").rstrip(self.img_ext)
                    # f.write(name + "\t" + file1.split(".")[0]+ "\t")#.lstrip("0").rstrip(self.img_ext)
                    # print(name + "\t" + file1.split(".")[0] + "\t" + other_dir + "\t" + file2.split(".")[0] + "\t")
                    # print("\n")
                # f.write("\n")


if __name__ == '__main__':
    data_dir = "/mnt/200230d3-9923-463d-8b4f-44d5439b890b/Data/images_I/"
    # data_dir = "/mnt/200230d3-9923-463d-8b4f-44d5439b890b/celebA_Data/"
    pairs_filepath = "pairs/pairs1.txt"
    img_ext = ".png"
    # a = os.listdir(data_dir)

    generatePairs = GeneratePairs(data_dir, pairs_filepath, img_ext)
    generatePairs.generate()
