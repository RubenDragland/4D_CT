import torch
import h5py
import sys, os, argparse, shutil
import numpy as np
import tqdm
from models import Generator3DTomoGAN
import matplotlib.pyplot as plt
import torchio as tio
import utils


def enhance_mssim(
    model_path,
    model_name,
    data_folder,
    data_name,
    key_input,
    key_target="gt",
    focus=None,
    sl=128,
    stride=128,
):
    # Load model
    model = Generator3DTomoGAN()
    model.load_state_dict(torch.load(os.path.join(model_path, f"{model_name}.pth")))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load data

    with h5py.File(os.path.join(data_folder, f"{data_name}.h5"), "r+") as data:
        # Visualisation
        # items = list(data[key_input].keys()).__len__()
        items = 1
        sl = sl

        # RSD: While debugging.
        if sl == stride:
            try:
                del data[f"{key_input}_enhanced"]
            except:
                pass

            try:
                del data[f"{key_input}_enhanced_{focus[0]}{focus[1]}{focus[2]}"]
            except:
                pass

        for i in tqdm.tqdm(range(items)):
            if focus is not None:
                if items == 1:
                    rec = torch.from_numpy(np.array(data[key_input]))[
                        focus[0, 0] : focus[0, 1],
                        focus[1, 0] : focus[1, 1],
                        focus[2, 0] : focus[2, 1],
                    ]
                    gt = torch.from_numpy(np.array(data[key_target]))[
                        focus[0, 0] : focus[0, 1],
                        focus[1, 0] : focus[1, 1],
                        focus[2, 0] : focus[2, 1],
                    ]
                else:
                    rec = torch.from_numpy(np.array(data[key_input][str(i).zfill(5)]))[
                        focus[0, 0] : focus[0, 1],
                        focus[1, 0] : focus[1, 1],
                        focus[2, 0] : focus[2, 1],
                    ]
                    gt = torch.from_numpy(np.array(data[key_target][str(i).zfill(5)]))[
                        focus[0, 0] : focus[0, 1],
                        focus[1, 0] : focus[1, 1],
                        focus[2, 0] : focus[2, 1],
                    ]

            else:
                if items == 1:
                    rec = torch.from_numpy(np.array(data[key_input]))
                    gt = torch.from_numpy(np.array(data[key_target]))
                else:
                    rec = torch.from_numpy(np.array(data[key_input][str(i).zfill(5)]))
                    gt = torch.from_numpy(np.array(data[key_target][str(i).zfill(5)]))

            rec = rec.unsqueeze(0)
            # rec = tio.RescaleIntensity((0, 1))(rec.unsqueeze(0)).to(
            #     device
            # )  # RSD: Difference here.
            rec_enhanced = torch.zeros_like(rec).to(device)
            # RSD: NB! Not ready for other stride.

            sizes = rec.shape
            print(sizes)
            a = sl  # Hard coded.
            x = np.arange(0, sizes[1], stride)
            y = np.arange(0, sizes[2], stride)
            z = np.arange(0, sizes[3], stride)

            # assert (
            #     x[-1] + a <= sizes[1]
            #     and y[-1] + a <= sizes[2]
            #     and z[-1] + a <= sizes[3]
            # )
            x = x[np.where(x + a <= sizes[1])]
            y = y[np.where(y + a <= sizes[2])]
            z = z[np.where(z + a <= sizes[3])]

            X, Y, Z = np.meshgrid(x, y, z)
            X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()

            ssims = np.zeros(len(X))
            psnrs = np.zeros(len(X))
            old_ssims = np.zeros(len(X))
            old_psnrs = np.zeros(len(X))
            coords = np.zeros((len(X), 3))

            for j, (xj, yj, zj) in tqdm.tqdm(enumerate(zip(X, Y, Z))):
                rec_dv = tio.RescaleIntensity((0, 1))(
                    rec[:, xj : xj + a, yj : yj + a, zj : zj + a]
                ).to(device)
                # RSD: This solution or to enhance all and then normalise.

                gt_sl = tio.RescaleIntensity((0, 1))(
                    gt[xj : xj + a, yj : yj + a, zj : zj + a].unsqueeze(0).to("cpu")
                ).to("cpu")

                rec_sl = tio.RescaleIntensity((0, 1))(
                    rec[:, xj : xj + a, yj : yj + a, zj : zj + a].to("cpu")
                ).to("cpu")

                # Enhance

                with torch.no_grad():
                    # RSD: slice not normalised before input.
                    rec_enhanced[
                        :, xj : xj + a, yj : yj + a, zj : zj + a
                    ] = model.forward(rec_dv.unsqueeze(0)).unsqueeze(0)

                    rec_enhanced_sl = tio.RescaleIntensity((0, 1))(
                        rec_enhanced[:, xj : xj + a, yj : yj + a, zj : zj + a].to("cpu")
                    )

                    ssims[j] = (
                        utils.torch_ssim(gt_sl, rec_enhanced_sl).detach().cpu().numpy()
                    )
                    psnrs[j] = (
                        utils.torch_psnr(gt_sl, rec_enhanced_sl).detach().cpu().numpy()
                    )

                    old_ssims[j] = (
                        utils.torch_ssim(gt_sl, rec_sl).detach().cpu().numpy()
                    )
                    old_psnrs[j] = (
                        utils.torch_psnr(gt_sl, rec_sl).detach().cpu().numpy()
                    )

                    coords[j] = np.array([xj, yj, zj])

                # Save
            if items == 1:
                data.create_dataset(
                    f"{key_input}_enhanced",
                    data=np.squeeze(rec_enhanced.detach().cpu().numpy()),
                )
            else:
                data.create_group(f"{key_input}_enhanced")
                data[f"{key_input}_enhanced"].create_dataset(
                    str(i).zfill(5),
                    data=np.squeeze(rec_enhanced.detach().cpu().numpy()),
                )

            # np.save(f"{key_input}_coords.npy", coords)
            # np.save(f"{key_input}_ssims.npy", ssims)
            # np.save(f"{key_input}_psnrs.npy", psnrs)
            # np.save(f"{key_input}_old_ssims.npy", old_ssims)
            # np.save(f"{key_input}_old_psnrs.npy", old_psnrs)

            np.savez(
                f"{data_name}_{key_input}_enhancement_metrics.npz",
                coords=coords,
                ssims=ssims,
                psnrs=psnrs,
                old_ssims=old_ssims,
                old_psnrs=old_psnrs,
                a=np.array([a]),
            )

            print(f"MSSIM: {np.mean(ssims)}, MPSNR: {np.mean(psnrs)}")

            print(f"OLD | SSIM: {np.mean(old_ssims)}, MPSNR: {np.mean(old_psnrs)}")

            del rec_enhanced

            del rec, gt

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
    parser.add_argument("-keyInput", type=str, required=True, help="keyInput")
    parser.add_argument(
        "-keyTarget", type=str, required=False, default="gt", help="keyTarget"
    )
    parser.add_argument(
        "-focus", type=int, required=False, nargs="+", default=0, help="Centre RoI"
    )
    parser.add_argument("-sl", type=int, required=False, default=128, help="Kernel dim")
    parser.add_argument("-stride", type=int, required=False, default=128, help="Stride")

    args, unparsed = parser.parse_known_args()

    # Run Enhancement

    if args.focus == 0:
        focus = None
    else:
        f = list(args.focus)
        focus = np.array([[f[0], f[1]], [f[2], f[3]], [f[4], f[5]]])

    enhance_mssim(
        args.modelPath,
        args.modelName,
        args.dataFolder,
        args.dataName,
        args.keyInput,
        key_target=args.keyTarget,
        focus=focus,
        sl=args.sl,
        stride=args.stride,
    )
