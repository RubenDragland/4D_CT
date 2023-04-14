import torch
import h5py
import sys, os, argparse, shutil
import numpy as np
import tqdm
from models import Generator3DTomoGAN
import matplotlib.pyplot as plt
import torchio as tio


def enhance(model_path, model_name, data_folder, data_name):

    # Load model
    model = Generator3DTomoGAN()
    model.load_state_dict(torch.load(os.path.join(model_path, f"{model_name}.pth")))
    model.eval()
    model.to("cpu")

    # print(model)

    key_name = "noisy3D"

    # Load data

    with h5py.File(os.path.join(data_folder, f"{data_name}.h5"), "r+") as data:

        # data.create_group("Enhanced")

        # Visualisation
        items = list(data[key_name].keys()).__len__()
        items = 1
        sl = 256

        fig, ax = plt.subplots(4, items, figsize=(15, 5 * items))

        # Give name to folder.
        # Consider parallelisation
        # for i in range(items):  # tqdm.tqdm(range(len(data[key_name]))):
        #     # Get data

        #     rec = torch.from_numpy(np.array(data[key_name][str(i).zfill(5)]))[
        #         sl:-sl, sl:-sl, sl:-sl
        #     ]

        #     rec = tio.RescaleIntensity((0, 1))(rec.unsqueeze(0))

        #     print("rec.shape", rec.shape)

        #     # Enhance

        #     with torch.no_grad():
        #         rec_enhanced = torch.squeeze(model.forward(rec.unsqueeze(0)))

        #     print("Done2")
        #     # Save

        #     # data["Enhanced"].create_dataset(str(i).zfill(5), data=rec_enhanced)

        #     # Visualisation
        #     rec = torch.squeeze(rec)
        #     idx = rec.shape[0] // 2

        #     ax[0, i].imshow(rec[idx, :, :], cmap="gray")
        #     ax[1, i].imshow(rec[:, :, idx], cmap="gray")
        #     ax[2, i].imshow(rec_enhanced[idx, :, :], cmap="gray")
        #     ax[3, i].imshow(rec_enhanced[:, :, idx], cmap="gray")

        #     print("Done3")

        for i in tqdm.tqdm(range(items)):

            rec = torch.from_numpy(np.array(data[key_name][str(i).zfill(5)]))
            # [
            #     sl:-sl, sl:-sl, sl:-sl
            # ]
            rec = tio.RescaleIntensity((0, 1))(rec.unsqueeze(0))
            rec_enhanced = torch.zeros_like(rec)

            sizes = rec.shape
            a = 256  # Hard coded.
            x = np.arange(0, sizes[1], a)
            y = np.arange(0, sizes[2], a)
            z = np.arange(0, sizes[3], a)

            assert (
                x[-1] + a <= sizes[1]
                and y[-1] + a <= sizes[2]
                and z[-1] + a <= sizes[3]
            )

            X, Y, Z = np.meshgrid(x, y, z)
            X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()

            for j, (xj, yj, zj) in tqdm.tqdm(enumerate(zip(X, Y, Z))):

                rec_dv = rec[:, xj : xj + a, yj : yj + a, zj : zj + a]

                # Enhance

                with torch.no_grad():
                    rec_enhanced[
                        :, xj : xj + a, yj : yj + a, zj : zj + a
                    ] = model.forward(rec_dv.unsqueeze(0)).unsqueeze(0)

                # Save

                # data["Enhanced"].create_dataset(str(i).zfill(5), data=rec_enhanced)

            # Visualisation
            rec = torch.squeeze(rec)
            rec_enhanced = torch.squeeze(rec_enhanced)
            idx = rec.shape[0] // 2

            ax[0].imshow(rec[idx, :, :], cmap="gray")
            ax[1].imshow(rec[:, :, idx], cmap="gray")
            ax[2].imshow(rec_enhanced[idx, :, :], cmap="gray")
            ax[3].imshow(rec_enhanced[:, :, idx], cmap="gray")

    plt.show()
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="3DTomoGAN, load trained model and enhance reconstruction"
    )

    parser.add_argument(
        "-modelPath", type=str, required=True, help="path to model folder"
    )
    parser.add_argument("-modelName", type=str, required=True, help="name of model")
    parser.add_argument("-dataFolder", type=str, required=True, help="Rec folder")
    parser.add_argument("-dataName", type=str, required=True, help="Rec name")

    args, unparsed = parser.parse_known_args()

    # Run Enhancement

    enhance(args.modelPath, args.modelName, args.dataFolder, args.dataName)
