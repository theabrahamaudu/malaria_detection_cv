import os
import cv2
from tqdm import tqdm
from src.utils.file_utilities import scan_directory


def convert_to_jpg():
    raw_path = './data/raw/SpeciesDataset/'
    raw_train_path = 'train/'
    raw_test_path = 'test/'
    species = ['Falciparum', 'Malariae', 'Ovale', 'Vivax']
    interim_path = './data/interim/'

    for split in [raw_train_path, raw_test_path]:
        for specie in species:
            os.makedirs(interim_path + specie, exist_ok=True)
            image_paths = scan_directory(raw_path + split + specie, '.png')
            for image in tqdm(image_paths,
                              total=len(image_paths),
                              desc="Converting %s images to jpg" % specie,
                              unit="images"):
                png_img = cv2.imread(raw_path + split + specie + '/' + image)
                cv2.imwrite(
                    interim_path + specie + '/' + image[:-4] + '.jpg',
                    png_img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                )


if __name__ == '__main__':
    convert_to_jpg()
