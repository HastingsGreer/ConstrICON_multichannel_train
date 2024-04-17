import itk
import os
from datetime import datetime

import footsteps
import numpy as np
import torch
import torch.nn.functional as F

import icon_registration as icon
import icon_registration.network_wrappers as network_wrappers
import icon_registration.networks as networks
from icon_registration import config
from icon_registration.losses import ICONLoss, to_floats
from icon_registration.mermaidlite import compute_warped_image_multiNC
import icon_registration.itk_wrapper



from train_brain import make_net

def get_model():
    net = make_net()
    from os.path import exists
    weights_location = "network_weights/network_weights_27000"
    if not exists(weights_location):
        print("Downloading pretrained model")
        import urllib.request
        import os
        download_path = "https://github.com/HastingsGreer/ConstrICON_multichannel_train/releases/download/weights/network_weights_27000"
        os.makedirs("network_weights/", exist_ok=True)
        urllib.request.urlretrieve(download_path, weights_location)
    trained_weights = torch.load(weights_location, map_location=torch.device("cpu"))
    net.regis_net.load_state_dict(trained_weights)
    net.to(config.device)
    net.eval()
    return net

def preprocess(image):
    max_ = np.max(np.array(image))

    arv = itk.GetArrayViewFromImage(image)
    arv /= max_

    return image

def to_image_list(image):
    arr = itk.array_from_image(image)

    ret = []

    for i in range(4):
        im = itk.image_from_array(arr[i])
        im.SetOrigin(np.array(image.GetOrigin())[:3])
        im.SetSpacing(np.array(image.GetSpacing())[:3])
        im.SetDirection(np.array(image.GetDirection())[:3, :3])
        ret.append(im)
    return ret

if __name__ == "__main__":
    import itk
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed")
    parser.add_argument("--moving")
    parser.add_argument("--transform_out")
    parser.add_argument("--warped_moving_out", default=None)
    parser.add_argument("--spacing_debug", action='store_true')
    parser.add_argument("--io_iterations", required=False,
                         default="None", help="The number of IO iterations. Default is None.")

    args = parser.parse_args()

    if args.io_iterations == "None":
        io_iterations = None
    else:
        io_iterations = int(args.io_iterations)

    net = get_model()

    fixed = itk.imread(args.fixed)
    moving = itk.imread(args.moving)

    fixed = preprocess(fixed)
    moving = preprocess(moving)

    if args.spacing_debug:
        print(fixed.GetSpacing())
        print(fixed.GetDirection())
        print(fixed.GetOrigin())
        print(fixed.GetLargestPossibleRegion().GetSize())

    fixed = to_image_list(fixed)
    moving = to_image_list(moving)

    if args.spacing_debug:
        print(fixed[0].GetSpacing())
        print(fixed[0].GetDirection())
        print(fixed[0].GetOrigin())
        print(fixed[0].GetLargestPossibleRegion().GetSize())


    phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair_with_multimodalities(net,moving, fixed, finetune_steps=io_iterations)

    itk.transformwrite([phi_AB], args.transform_out)

    if args.warped_moving_out and args.spacing_debug:
        interpolator = itk.LinearInterpolateImageFunction.New(moving[0])
        warped_image_A = itk.resample_image_filter(
                moving[0],
                transform=phi_AB,
                interpolator=interpolator,
                use_reference_image=True,
                reference_image=fixed[0]
                )
        itk.imwrite(warped_image_A, args.warped_moving_out)
    if args.spacing_debug:
        itk.imwrite(fixed[0], "fixed_processed.mha")
        itk.imwrite(moving[0], "moving_processed.mha")





