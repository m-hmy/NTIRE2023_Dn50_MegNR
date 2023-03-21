import os.path
import logging
import torch
import argparse
import json
import glob
import numpy as  np
import random
import gc

from pprint import pprint
from utils.model_summary import get_model_activation, get_model_flops
from utils import utils_logger
from utils import utils_image as util


def select_model(args, device):
    # Model ID is assigned according to the order of the submissions.
    # Different networks are trained with input range of either [0,1] or [0,255]. The range is determined manually.
    model_id = args.model_id
    if model_id == 3:
        # SGN test
        from models.team20_megnr import HAUformer
        name, data_range = f"{model_id:02}_megnr_U_HAT_large", 1.0
        model_path = os.path.join('model_zoo', 'team20_megnr_v2.pth')
        model = HAUformer()

        state_dict = torch.load(model_path, map_location="cpu")["hau"]
        model.load_state_dict(state_dict, strict=True)
    elif model_id == 1:
        from models.team20_megnr import Restormer
        name, data_range = f"{model_id:02}_megnr_restormer_arch", 1.0
        model_path = os.path.join('model_zoo', 'team20_megnr_v2.pth')
        model = Restormer()

        state_dict = torch.load(model_path)["restormer"]["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.find("model.") >= 0:
                new_state_dict[k.replace("model.", "")] = v
        model.load_state_dict(state_dict, strict=True)
    elif model_id == 2:
        from models.team20_megnr import KBNet_s
        name, data_range = f"{model_id:02}_megnr_kbnet_arch", 1.0
        model_path = os.path.join('model_zoo', 'team20_megnr_v2.pth')
        model = KBNet_s()

        state_dict = torch.load(model_path)["kbnet"]
        # state_dict.pop("current_val_metric")
        # state_dict.pop("best_val_metric")
        # state_dict.pop("best_iter")
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if k.find("model.") >= 0:
        #         new_state_dict[k.replace("model.", "")] = v
        model.load_state_dict(state_dict, strict=True)
    # elif model_id == 3:
    #     from models.megnr_ensemble_model import KBNet_s
    #     name, data_range = f"{model_id:02}_megnr_kbnet_arch", 1.0
    #     model_path = os.path.join('model_zoo', 'team00_sgn.ckpt')
    #     model = KBNet_s()
    elif model_id == 0:
        from models.team20_megnr import Unet, GaussianDiffusion, MIDPM
        name, data_range = f"{model_id:02}_megnr_midpm", 1.0
        model_path = os.path.join('model_zoo', 'team20_megnr_v2.pth')
        model_unet = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()

        diffusion = GaussianDiffusion(
            model_unet,
            image_size=256,
            timesteps=1000,  # number of steps
            sampling_timesteps=
            10,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
            loss_type='l1',  # L1 or L2
            objective='pred_x0',
            # p2_loss_weight_gamma=1.,
        )

        model = MIDPM(
            diffusion,
            ema_decay=0.995,  # exponential moving average decay
        )
        model.load(model_path)
        tile = None

        return model, name, data_range, tile

    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    # print(model)
    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    return model, name, data_range, tile


def select_dataset(data_dir, mode):
    if mode == "test":
        path = [
            (
                os.path.join(data_dir, f"DIV2K_test_noise50/{i:04}.png"),
                os.path.join(data_dir, f"DIV2K_test_HR/{i:04}.png")
            ) for i in range(901, 1001)
        ]
        # [f"DIV2K_test_LR/{i:04}.png" for i in range(901, 1001)]
    elif mode == "valid":
        path = [
            (
                os.path.join(data_dir, f"DIV2K_valid_noise50/{i:04}.png"),
                os.path.join(data_dir, f"DIV2K_valid_HR/{i:04}.png")
            ) for i in range(801, 901)
        ]
    elif mode == "hybrid_test":
        path = [
            (
                p.replace("_HR", "_LR").replace(".png", "noise50.png"),
                p
            ) for p in sorted(glob.glob(os.path.join(data_dir, "LSDIR_DIV2K_test_HR/*.png")))
        ]
    else:
        raise NotImplementedError(f"{mode} is not implemented in select_dataset")
    return path


def forward(img_lq, model, tile=None, tile_overlap=32, scale=1):
    if tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        tile_overlap = tile_overlap
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

def rotate(image, mode):
    '''
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformationss
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def rotate_re(image, mode):
    '''
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image, k=-1)
        # out = np.transpose(image, (1, 0, 2))
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.flipud(image)
        out = np.rot90(out, k=-1)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=-2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.flipud(image)
        out = np.rot90(out, k=-2)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=-3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.flipud(image)
        out = np.rot90(out, k=-3)
        
    else:
        raise Exception('Invalid choice of image transformation')

    return out


def inference(net, img, pch_size=256, data_range=1, model_name = ""):
    """
    model inference
    """

    nndenoise_model = net
    # img = img[...,None]
    overlap_size = 32
    init_img = img
    img_list = []
    for i in range(8):
        img = rotate(init_img, i)
        h,w = img.shape[:2]
        overlap_size = random.randint(8, 32)
        ind_H = list(range(0, h-pch_size,pch_size - overlap_size))
        if ind_H[-1]<h-pch_size:
            ind_H.append(h-pch_size)
        ind_W = list(range(0, w - pch_size, pch_size - overlap_size))
        if ind_W[-1]<w - pch_size:
            ind_W.append(w-pch_size)

        img_out = np.zeros([h , w , 3])
        mask_out = np.zeros([h , w , 3])
        for start_H in ind_H:
            for start_W in ind_W:
                # pch_noisy = img[start_H:start_H + pch_size, start_W:start_W+pch_size, ...]
                img_patch = img[start_H:start_H + pch_size, start_W:start_W+pch_size, ...]
                # pch_in = img_patch[None,...]
                # pch_in = np.transpose(pch_in, (0, 3, 1, 2))
                # pch_in = torch.FloatTensor(pch_in.copy()).cuda()
                pch_in = util.uint2tensor4(img_patch, data_range)
                pch_in = pch_in.cuda()
                # pch_restored = nndenoise_model(pch_in).cpu().detach().numpy()
                # print(pch_in.shape)
                pch_out = []
                for n in range(1):

                    pch_in_temp = pch_in +  torch.randn(pch_in.shape).cuda()/255
                    # pch_in_temp = pch_in
                    # pch_in_temp = torch.FloatTensor(pch_in_temp.copy()).cuda()  # .copy get a contiguous array
                    # print(pch_in.shape)
                    if "midpm" in model_name:
                        pch_restored_temp = nndenoise_model(pch_in_temp, 730).cpu().detach().numpy()
                    else:
                        pch_restored_temp = nndenoise_model(pch_in_temp).cpu().detach().numpy()
                    print(pch_in_temp.shape, pch_restored_temp.shape)
                    pch_out.append(pch_restored_temp)
                    # print(pch_restored.shape)
                # pch_out = []
                pch_out = np.mean(np.stack(pch_out, 0),0)
                pch_restored = pch_out[0]
                # pch_restored = pch_restored[0]
                
                pch_restored = np.transpose(pch_restored,(1, 2, 0))
                img_out[start_H:start_H + pch_size, start_W:start_W+pch_size, ...] += pch_restored
                mask_out[start_H:start_H + pch_size, start_W:start_W+pch_size, ...] += np.ones([pch_size, pch_size,3])
        img_out = (img_out/mask_out*255.0).clip(0, 255)
        img_out = rotate_re(img_out, i)
        img_list.append(img_out)
    return np.mean(np.stack(img_list, axis=0), axis=0) 


def run(model, model_name, data_range, tile, logger, device, args, mode="test"):

    sf = 4
    border = sf
    results = dict()
    results[f"{mode}_runtime"] = []
    results[f"{mode}_psnr"] = []
    if args.ssim:
        results[f"{mode}_ssim"] = []
    # results[f"{mode}_psnr_y"] = []
    # results[f"{mode}_ssim_y"] = []

    # --------------------------------
    # dataset path
    # --------------------------------
    data_path = select_dataset(args.data_dir, mode)
    save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    print(model_name)

    for i, (img_noisy, img_hr) in enumerate(data_path):
        # print(img_noisy)
        # print(img_hr)
        # --------------------------------
        # (1) img_noisy
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_hr))
        img_noisy = util.imread_uint(img_noisy, n_channels=3)
        # print(img_noisy.shape)
        # img_noisy = util.uint2tensor4(img_noisy, data_range)
        # img_noisy = img_noisy.to(device)

        # --------------------------------
        # (2) img_dn
        # --------------------------------
        start.record()
        # img_dn = forward(img_noisy, model, tile)
        img_dn = inference(model, img_noisy, data_range = data_range, model_name = model_name)
        end.record()
        torch.cuda.synchronize()
        results[f"{mode}_runtime"].append(start.elapsed_time(end))  # milliseconds
        # img_dn = util.tensor2uint(img_dn, data_range)

        # --------------------------------
        # (3) img_hr
        # --------------------------------
        img_hr = util.imread_uint(img_hr, n_channels=3)
        img_hr = img_hr.squeeze()
        img_hr = util.modcrop(img_hr, sf)

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------

        # print(img_dn.shape, img_hr.shape)
        psnr = util.calculate_psnr(img_dn, img_hr, border=border)
        results[f"{mode}_psnr"].append(psnr)

        if args.ssim:
            ssim = util.calculate_ssim(img_dn, img_hr, border=border)
            results[f"{mode}_ssim"].append(ssim)
            logger.info("{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.".format(img_name + ext, psnr, ssim))
        else:
            logger.info("{:s} - PSNR: {:.2f} dB".format(img_name + ext, psnr))

        # if np.ndim(img_hr) == 3:  # RGB image
        #     img_dn_y = util.rgb2ycbcr(img_dn, only_y=True)
        #     img_hr_y = util.rgb2ycbcr(img_hr, only_y=True)
        #     psnr_y = util.calculate_psnr(img_dn_y, img_hr_y, border=border)
        #     ssim_y = util.calculate_ssim(img_dn_y, img_hr_y, border=border)
        #     results[f"{mode}_psnr_y"].append(psnr_y)
        #     results[f"{mode}_ssim_y"].append(ssim_y)
        # print(os.path.join(save_path, img_name+ext))
        util.imsave(img_dn, os.path.join(save_path, img_name+ext))

    results[f"{mode}_memory"] = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2
    results[f"{mode}_ave_runtime"] = sum(results[f"{mode}_runtime"]) / len(results[f"{mode}_runtime"]) #/ 1000.0
    results[f"{mode}_ave_psnr"] = sum(results[f"{mode}_psnr"]) / len(results[f"{mode}_psnr"])
    if args.ssim:
        results[f"{mode}_ave_ssim"] = sum(results[f"{mode}_ssim"]) / len(results[f"{mode}_ssim"])
    # results[f"{mode}_ave_psnr_y"] = sum(results[f"{mode}_psnr_y"]) / len(results[f"{mode}_psnr_y"])
    # results[f"{mode}_ave_ssim_y"] = sum(results[f"{mode}_ssim_y"]) / len(results[f"{mode}_ssim_y"])
    logger.info("{:>16s} : {:<.3f} [M]".format("Max Memery", results[f"{mode}_memory"]))  # Memery
    logger.info("------> Average runtime of ({}) is : {:.6f} seconds".format("test" if mode == "test" else "valid", results[f"{mode}_ave_runtime"]))

    return results


def main(args):

    utils_logger.logger_info("NTIRE2023-Dn50", log_path="NTIRE2023-Dn50.log")
    logger = logging.getLogger("NTIRE2023-Dn50")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    save_path_list = []
    temp_mode = ""
    for ei in range(4):
        args.model_id = ei
        model, model_name, data_range, tile = select_model(args, device)
        
        logger.info(model_name)

        # if model not in results:
        if True:
            # --------------------------------
            # restore image
            # --------------------------------
            

            if args.hybrid_test:
                # inference on the DIV2K and LSDIR test set
                valid_results = run(model, model_name, data_range, tile, logger, device, args, mode="hybrid_test")
                # record PSNR, runtime
                results[model_name] = valid_results
                temp_mode="hybrid_test"
            else:
                # inference on the validation set
                valid_results = run(model, model_name, data_range, tile, logger, device, args, mode="valid")
                # record PSNR, runtime
                results[model_name] = valid_results
                temp_mode="valid"
                if args.include_test:
                    # inference on the test set
                    test_results = run(model, model_name, data_range, tile, logger, device, args, mode="test")
                    
                    results[model_name].update(test_results)
                    temp_mode="test"


            save_path_list.append(os.path.join(args.save_dir, model_name, temp_mode))
            if "midpm" in model_name:
                # del model
                # gc.collect()
                # torch.cuda.empty_cache()
                # continue
                model = model.model.model
                input_dim = {"x":torch.FloatTensor(1, 3, 256, 256).cuda(), "time":torch.Tensor(1).cuda()}  # set the input dimension
                activations, num_conv = get_model_activation(model, input_dim, dict)
                activations = activations/10**6
                logger.info("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
                logger.info("{:>16s} : {:<d}".format("#Conv2d", num_conv))

                flops = get_model_flops(model, input_dim, False, dict)
                flops = flops/10**9
                logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

                num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
                num_parameters = num_parameters/10**6
                logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
                results[model_name].update({"activations": activations, "num_conv": num_conv, "flops": flops, "num_parameters": num_parameters})
                del model, input_dim
                gc.collect()
                torch.cuda.empty_cache()
                continue

            input_dim = (3, 256, 256)  # set the input dimension
            activations, num_conv = get_model_activation(model, input_dim)
            activations = activations/10**6
            logger.info("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
            logger.info("{:>16s} : {:<d}".format("#Conv2d", num_conv))

            flops = get_model_flops(model, input_dim, False)
            flops = flops/10**9
            logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

            num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
            num_parameters = num_parameters/10**6
            logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
            results[model_name].update({"activations": activations, "num_conv": num_conv, "flops": flops, "num_parameters": num_parameters})
            if(ei!=2):
                del model
                gc.collect()
                torch.cuda.empty_cache()

        
    # temp_mode = ""
    if args.hybrid_test:
        temp_mode = "hybrid_test"
    else:
        temp_mode = "valid"
        if args.include_test:
            temp_mode = "test"

    model_name_final = "megnr_ensemble"
    results[model_name_final] = results[model_name]
    save_path_final = os.path.join(args.save_dir, model_name_final, temp_mode)
    util.mkdir(save_path_final)
    data_path=select_dataset(args.data_dir, temp_mode)

    results[model_name_final][f"{temp_mode}_psnr"] = []
    if args.ssim:
        results[model_name_final][f"{temp_mode}_ssim"] = []
    for i, (_, img_hr) in enumerate(data_path):
        # print(img_noisy)
        # print(img_hr)
        # --------------------------------
        # (1) img_noisy
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_hr))
        img_list = []
        for ei in range(4):
            img_res_path = os.path.join(save_path_list[ei], img_name + ext) 
            img_res = util.imread_uint(img_res_path, n_channels=3)
            img_res = img_res.squeeze().astype(np.float)
            img_res = util.modcrop(img_res, 4)
            img_list.append(img_res)
        img_dn = np.mean(img_list, axis=0)
        img_hr = util.imread_uint(img_hr, n_channels=3)
        img_hr = img_hr.squeeze()
        img_hr = util.modcrop(img_hr, 4)
        psnr = util.calculate_psnr(img_dn, img_hr, border=4)
        results[model_name_final][f"{temp_mode}_psnr"].append(psnr)

        if args.ssim:
            ssim = util.calculate_ssim(img_dn, img_hr, border=4)
            results[f"{temp_mode}_ssim"].append(ssim)
            logger.info("{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.".format(img_name + ext, psnr, ssim))
        else:
            logger.info("{:s} - PSNR: {:.2f} dB".format(img_name + ext, psnr))

        util.imsave(img_dn, os.path.join(save_path_final, img_name+ext))
    results[model_name_final][f"{temp_mode}_ave_psnr"] = sum(results[model_name_final][f"{temp_mode}_psnr"]) / len(results[model_name_final][f"{temp_mode}_psnr"])
    if args.ssim:
        results[model_name_final][f"{temp_mode}_ave_ssim"] = sum(results[model_name_final][f"{temp_mode}_ssim"]) / len(results[model_name_final][f"{temp_mode}_ssim"])
    
    with open(json_dir, "w") as f:
            json.dump(results, f)

    if args.include_test:
        fmt = "{:20s}\t{:10s}\t{:10s}\t{:14s}\t{:14s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Test PSNR", "Val Time [ms]", "Test Time [ms]", "Ave Time [ms]",
                       "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    else:
        fmt = "{:20s}\t{:10s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Val Time [ms]", "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    for k, v in results.items():
        # print(v.keys())
        if args.hybrid_test:
            val_psnr = f"{v['hybrid_test_ave_psnr']:2.2f}"
            val_time = f"{v['hybrid_test_ave_runtime']:3.2f}"
            mem = f"{v['hybrid_test_memory']:2.2f}"
        else:
            val_psnr = f"{v['valid_ave_psnr']:2.2f}"
            val_time = f"{v['valid_ave_runtime']:3.2f}"
            mem = f"{v['valid_memory']:2.2f}"
        num_param = f"{v['num_parameters']:2.3f}"
        flops = f"{v['flops']:2.2f}"
        acts = f"{v['activations']:2.2f}"
        conv = f"{v['num_conv']:4d}"
        if args.include_test:
            # from IPython import embed; embed()
            test_psnr = f"{v['test_ave_psnr']:2.2f}"
            test_time = f"{v['test_ave_runtime']:3.2f}"
            ave_time = f"{(v['valid_ave_runtime'] + v['test_ave_runtime']) / 2:3.2f}"
            s += fmt.format(k, val_psnr, test_psnr, val_time, test_time, ave_time, num_param, flops, acts, mem, conv)
        else:
            s += fmt.format(k, val_psnr, val_time, num_param, flops, acts, mem, conv)
    with open(os.path.join(os.getcwd(), 'results.txt'), "w") as f:
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2023-Dn50")
    parser.add_argument("--data_dir", default="/cluster/work/cvl/yawli/data/NTIRE2023_Challenge", type=str)
    parser.add_argument("--save_dir", default="/cluster/work/cvl/yawli/data/NTIRE2023_Challenge/results", type=str)
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--include_test", action="store_true", help="Inference on the DIV2K test set")
    parser.add_argument("--hybrid_test", action="store_true", help="Hybrid test on DIV2K and LSDIR test set")
    parser.add_argument("--ssim", action="store_true", help="Calculate SSIM")

    args = parser.parse_args()
    pprint(args)

    main(args)
