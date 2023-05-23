import torch
import h5py
import sys, os, argparse, shutil
import numpy as np
import tqdm
from models import Generator3DTomoGAN
import matplotlib.pyplot as plt
import torchio as tio
import utils


def enhance(
    model_path,
    model_name,
    data_folder,
    data_name,
    key_input,
    key_target="gt",
    focus=0,
    dims=0,
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

        # RSD: While debugging.

        try:
            del data[f"{key_input}_enhanced"]
        except:
            pass

        try:
            del data[f"{key_input}_enhanced_{focus[0]}{focus[1]}{focus[2]}"]
        except:
            pass

        # fig, ax = plt.subplots(4, items, figsize=(15, 5 * items))

        if focus:
            if dims == 0:
                ax, ay, az = 256, 256, 256
            else:
                ax, ay, az = dims

            rec = torch.from_numpy(np.array(data[key_input]))
            xm, ym, zm = np.array(focus)
            rec = rec[
                xm - ax // 2 : xm + ax // 2,
                ym - ay // 2 : ym + ay // 2,
                zm - az // 2 : zm + az // 2,
            ]
            rec = tio.RescaleIntensity((0, 1))(rec.unsqueeze(0)).to(device)
            rec_enhanced = torch.zeros_like(rec).to(device)
            print(rec.shape)

            with torch.no_grad():
                rec_enhanced = model.forward(rec.unsqueeze(0)).unsqueeze(0)

            data.create_dataset(
                f"{key_input}_enhanced_{focus[0]}{focus[1]}{focus[2]}",
                data=np.squeeze(rec_enhanced.detach().cpu().numpy()),
            )

            gt = torch.from_numpy(np.array(data[key_target]))
            gt = gt[
                xm - ax // 2 : xm + ax // 2,
                ym - ay // 2 : ym + ay // 2,
                zm - az // 2 : zm + az // 2,
            ]
            gt = tio.RescaleIntensity((0, 1))(gt.unsqueeze(0)).to(device)
            rec_enhanced = tio.RescaleIntensity((0, 1))(
                torch.squeeze(rec_enhanced.detach().cpu()).unsqueeze(0)
            ).to(device)

            print(gt.shape, rec_enhanced.shape)

            ssim = utils.torch_ssim(gt, rec_enhanced).detach().cpu().numpy()
            psnr = utils.torch_psnr(gt, rec_enhanced).detach().cpu().numpy()

            data.attrs[
                f"{key_input}_enhanced_ssim_{focus[0]}{focus[1]}{focus[2]}"
            ] = ssim
            data.attrs[
                f"{key_input}_enhanced_psnr_{focus[0]}{focus[1]}{focus[2]}"
            ] = psnr

            print(f"SSIM: {ssim}, PSNR: {psnr}")

            ssim = utils.torch_ssim(gt, rec).detach().cpu().numpy()
            psnr = utils.torch_psnr(gt, rec).detach().cpu().numpy()

            print(f"OLD | SSIM: {ssim}, PSNR: {psnr}")

            data.attrs[f"{key_input}_old_ssim_{focus[0]}{focus[1]}{focus[2]}"] = ssim
            data.attrs[f"{key_input}_old_psnr_{focus[0]}{focus[1]}{focus[2]}"] = psnr

        else:
            for i in tqdm.tqdm(range(items)):
                if items == 1:
                    rec = torch.from_numpy(np.array(data[key_input]))[:1536, :, :]
                else:
                    rec = torch.from_numpy(np.array(data[key_input][str(i).zfill(5)]))

                rec = tio.RescaleIntensity((0, 1))(rec.unsqueeze(0)).to(device)
                rec_enhanced = torch.zeros_like(rec).to(device)

                sizes = rec.shape
                print(sizes)
                a = 128  # Hard coded.
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
                    # RSD: NB! Not normalised input.

                    # Enhance

                    with torch.no_grad():
                        rec_enhanced[
                            :, xj : xj + a, yj : yj + a, zj : zj + a
                        ] = model.forward(rec_dv.unsqueeze(0)).unsqueeze(0)

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

                del rec, model

                # Determine PSNR and SSIM
                if items == 1:
                    gt = torch.from_numpy(np.array(data[key_target]))[:1536, :, :]
                else:
                    gt = torch.from_numpy(np.array(data[key_target][str(i).zfill(5)]))

                gt = tio.RescaleIntensity((0, 1))(gt.unsqueeze(0)).to("cpu")
                rec_enhanced = tio.RescaleIntensity((0, 1))(rec_enhanced.to("cpu"))

                ssim = utils.torch_ssim(gt, rec_enhanced).detach().cpu().numpy()
                psnr = utils.torch_psnr(gt, rec_enhanced).detach().cpu().numpy()

                print(f"SSIM: {ssim}, PSNR: {psnr}")

                if items == 1:
                    data.attrs[f"{key_input}_enhanced_ssim"] = ssim
                    data.attrs[f"{key_input}_enhanced_psnr"] = psnr
                else:
                    data[f"{key_input}_enhanced"].attrs["ssim"] = ssim
                    data[f"{key_input}_enhanced"].attrs["psnr"] = psnr
                    # RSD: Improve this. average or list or something.

                del rec_enhanced

                if items == 1:
                    rec = torch.from_numpy(np.array(data[key_input]))[:1536, :, :]
                else:
                    rec = torch.from_numpy(np.array(data[key_input][str(i).zfill(5)]))

                rec = tio.RescaleIntensity((0, 1))(rec.unsqueeze(0)).to("cpu")

                ssim = utils.torch_ssim(gt, rec).detach().cpu().numpy()
                psnr = utils.torch_psnr(gt, rec).detach().cpu().numpy()

                print(f"OLD | SSIM: {ssim}, PSNR: {psnr}")

                if items == 1:
                    data.attrs[f"{key_input}_old_ssim"] = ssim
                    data.attrs[f"{key_input}_old_psnr"] = psnr
                else:
                    data[f"{key_input}_old"].attrs["ssim"] = ssim
                    data[f"{key_input}_old"].attrs["psnr"] = psnr

                del rec, gt

        #         # Visualisation
        #         rec = torch.squeeze(rec).detach().cpu().numpy()
        #         rec_enhanced = torch.squeeze(rec_enhanced).detach().cpu().numpy()
        #         idx = rec.shape[0] // 2

        #         ax[0].imshow(rec[idx, :, :], cmap="gray")
        #         ax[1].imshow(rec[:, :, idx], cmap="gray")
        #         ax[2].imshow(rec_enhanced[idx, :, :], cmap="gray")
        #         ax[3].imshow(rec_enhanced[:, :, idx], cmap="gray")

        # plt.show()
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
    parser.add_argument(
        "-dims", type=int, required=False, nargs="+", default=0, help="Dimensions"
    )

    args, unparsed = parser.parse_known_args()

    # Run Enhancement

    enhance(
        args.modelPath,
        args.modelName,
        args.dataFolder,
        args.dataName,
        args.keyInput,
        key_target=args.keyTarget,
        focus=args.focus,
        dims=args.dims,
    )
