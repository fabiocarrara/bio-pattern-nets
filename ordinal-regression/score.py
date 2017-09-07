import os
import argparse
import torch

from scipy.misc import imread
from tqdm import tqdm
from model import Net, SmallNet, SmallDeepNet


def list_images(paths):
    image_list = []
    for path in paths:
        if os.path.isdir(path):
            image_list += [os.path.join(path, f)
                           for f in os.listdir(path)
                           if os.path.isfile(os.path.join(path, f))
                           and os.path.splitext(f)[1].lstrip('.').lower()
                           in ('jpg', 'jpeg', 'png')]
        elif os.path.isfile(path):
            image_list.append(path)
        else:
            print 'WARNING: {} not found, skipping'.format(path)

    return image_list


def image_generator(paths):
    for path in paths:
        img = imread(path)
        img = img.transpose((2, 0, 1)).astype(float)
        yield path, torch.from_numpy(img).float()


def batch_generator(images, batch_size=32):
    batch = []
    paths = []
    for path, image in images:
        paths.append(path)
        batch.append(image)
        if len(batch) == batch_size:
            yield paths, torch.stack(batch)
            paths[:] = []
            batch[:] = []  # clear the list

    if len(batch) > 0:
        yield paths, torch.stack(batch)


def main():
    parser = argparse.ArgumentParser(description='Score biological patterns images with a given model.')

    parser.add_argument('model', type=str, help='model snapshot')
    parser.add_argument('images', type=str, nargs='+', help='image files or folders containing images')

    parser.add_argument('-a', '--arch', type=str, default='Net', choices=['Net', 'SmallNet', 'SmallDeepNet'],
                        help='Model architecture [Net | SmallNet | SmallDeepNet] (default: Net)')
    parser.add_argument('-g', '--gpu', default=False, action='store_true', help='Use CUDA')
    parser.add_argument('-b', '--batchSize', type=int, default=32,
                        help='How many images to process in parallel, useful if GPU is available (default: 32)')
    args = parser.parse_args()

    # list all images
    images = list_images(args.images)
    # load images to tensors from filenames
    images = image_generator(images)
    # make batches of tensors
    batches = batch_generator(images, batch_size=args.batchSize)

    # instantiate the model
    model = eval(args.arch)()
    # transform the model from classification to regression
    model.to_ordinal()
    # move the model to the GPU if available
    if args.gpu:
        model.cuda()
    # load the model weights from file
    model.load_state_dict(torch.load(args.model))
    # set the model in evaluation mode
    model.eval()

    # for each batch
    for paths, batch in tqdm(batches):
        # move it to GPU if available
        if args.gpu:
            batch = batch.cuda()
        batch = torch.autograd.Variable(batch, volatile=True)
        # calculate scores
        scores = model(batch)
        # transform tensor of scores to list
        scores = scores.data.squeeze().tolist()
        # print out <imagepath, score> couples
        for path, score in zip(paths, scores):
            print '{}\t{}'.format(path, score)


if __name__ == '__main__':
    main()