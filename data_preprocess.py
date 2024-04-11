import itk
import numpy as np
import glob
import torch
import tqdm

output = []

for pth in tqdm.tqdm(sorted(glob.glob("data/degree_powers_normalized_dipy/degree_power_images/*"))[:-10]):
    print(pth)
    img = itk.imread(pth)
    img = np.array(img)
    print(img.dtype)
    img = torch.tensor(img)[None]
    output.append(img)

output = torch.cat(output)

print(output.shape)

import footsteps

torch.save(output, footsteps.output_dir + "train_tensor.trch")
