import pandas as pd
import os
from utils.visualization import *
from skimage.io import imread
from sklearn.model_selection import train_test_split
from utils.utilities import *


class DataLoader:

    def __init__(self, data_folder = "../data/"):
        self.test_images_path = data_folder + "images/test/"
        self.train_images_path = data_folder + "images/train/"

        train_images = set(os.listdir(self.train_images_path)) 
        # remove corrupted images
        train_images -= {'6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'}
        self.train_ship_segmentations_path = data_folder + "train_ship_segmentations_v2.csv"
        train_masks = pd.read_csv(self.train_ship_segmentations_path)
        train_masks = train_masks[(train_masks['ImageId'].isin(train_images))]

        train_masks['ships'] = train_masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
        self.train = train_masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
        self.train['has_ship'] = self.train['ships'].map(lambda x: 1.0 if x > 0 else 0.0)
        self.train['has_ship_vec'] = self.train['has_ship'].map(lambda x: [x])
        encodedPixelsDF = train_masks[['ImageId', 'EncodedPixels']].groupby('ImageId')['EncodedPixels'].apply(list)
        # gourped -- error in merge -- join on id
        self.train = self.train.join(encodedPixelsDF, 'ImageId')

    def get_masks(self, image_id, from_train=True):
        img_masks = None
        if from_train:
            img_masks = self.train.loc[self.train['ImageId'] == image_id, 'EncodedPixels'].values[0]
        else:
            print("ERROR: test masks are not loaded")

        # Take the individual ship masks and create a single mask array for all ships
        all_masks = np.zeros((768, 768))
        for mask in img_masks:
            all_masks += rle_decode(mask)
        return all_masks

    def get_grouped_images_gen(self, df=None, batch_size=8, img_scaling=(3, 3), f=None):
        if df is None:
            df = self.train
        out_rgb = []
        out_mask = []
        all_batches = list(df.groupby('ImageId'))
        while True:
            np.random.shuffle(all_batches)
            for c_img_id, c_masks in all_batches:
                c_img = self.get_image(c_img_id)
                c_mask = masks_as_image(c_masks['EncodedPixels'].values[0])
                if img_scaling is not None:
                    c_img = c_img[::img_scaling[0], ::img_scaling[1]]
                    c_mask = c_mask[::img_scaling[0], ::img_scaling[1]]
                if f is not None:
                    c_img = apply_filter(c_img, f)
                else:
                    c_img = c_img/255.0
                out_rgb += [c_img]
                out_mask += [c_mask]
                if len(out_rgb) >= batch_size:
                    yield np.stack(out_rgb, 0), np.stack(out_mask, 0)
                    out_rgb, out_mask = [], []

    def undersample_no_ships(self, frac=0.2):
        ships = self.train[self.train['has_ship'] == 1]
        no_ships = self.train[self.train['has_ship'] == 0]
        no_ships = no_ships.sample(frac=frac)
        return ships.append(no_ships)

    def oversample_multiships(self, times=3):
        multiships = self.train[self.train['ships'] > 1]
        df = multiships
        for _ in range(times-1):
            df = df.append(multiships)
        return df

    def adjust_set(self, frac=0.2, times=3):
        df1 = self.undersample_no_ships(frac)
        df2 = self.oversample_multiships(times)
        return df1.append(df2)

    def train_split(self, test_size=0.3, adjust_set=True, filterNan=False):
        df = self.train
        if adjust_set:
            df = self.adjust_set()
        if filterNan:
            df = df[df.has_ship == 1]
        train_df, valid_df = train_test_split(df, test_size=test_size, stratify=df['ships']) # the stratified class must be "ships"
        return train_df, valid_df

    def get_image(self, img_id, from_train=True):
        if from_train:
            return imread(self.train_images_path + img_id)
        else:
            return
