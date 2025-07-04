from segment_anything import sam_model_registry, SamPredictor
import torch.nn as nn
import torch
import argparse
import os
from torch import optim
from torch.utils.data import DataLoader
from DataLoader_adapt import TrainingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger, generate_point,generate_point1, setting_prompt_none
from GB import GazeWeightedCrossEntropyLoss
from GA import GALoss
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
import datetime
from torch.nn import functional as F
from apex import amp
import random
from sam_lora import LoRA_Sam
from torchvision.utils import save_image
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir/DDTI_exp", help="work dir")
    parser.add_argument("--run_name", type=str, default="sam-med2d", help="run model name")
    parser.add_argument("--epochs", type=int, default=25, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=1, help="get mask number")
    parser.add_argument("--data_path", type=str, default="data/DDTI", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume") 
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=None, help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--use_amp", type=bool, default=False, help="use amp")
    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def prompt_and_decoder_lora(args, batched_input, model, image_embeddings, decoder_iter = False):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None
    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.lora_vit.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )

    else:
        sparse_embeddings, dense_embeddings = model.lora_vit.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

    low_res_masks, iou_predictions,src = model.lora_vit.mask_decoder(
        image_embeddings = image_embeddings,
        image_pe = model.lora_vit.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )
  
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions,src
def prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None
    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )

    else:
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

    low_res_masks, iou_predictions,src = model.mask_decoder(
        image_embeddings = image_embeddings,
        image_pe = model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )
  
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions,src

def train_one_epoch(args, model,model_t, optimizer, train_loader, epoch, criterion,criterion_gaze,ga):
    train_loader = tqdm(train_loader)
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)
    for batch, batched_input in enumerate(train_loader):
            batched_input = stack_dict_batched(batched_input)
            batched_input = to_device(batched_input, args.device)
            
            if random.random() > 0.5:
                batched_input["point_coords"] = None
                flag = "boxes"
            else:
                batched_input["boxes"] = None
                flag = "point"

            for n, value in model.lora_vit.image_encoder.named_parameters():
                if "linear_a_q" in n or "linear_b_q" in n or "linear_a_v" in n or "linear_b_v" in n:
                    value.requires_grad = True
                else:
                    value.requires_grad = False

            if args.use_amp:
                labels = batched_input["label"].half()
                heatmap = batched_input["heatmap"].half()
                image_embeddings = model.lora_vit.image_encoder(batched_input["image"].half())
      
                batch, _, _, _ = image_embeddings.shape
                image_embeddings_repeat = []
                for i in range(batch):
                    image_embed = image_embeddings[i]
                    image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                    image_embeddings_repeat.append(image_embed)
                image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

                masks, low_res_masks, iou_predictions,src = prompt_and_decoder_lora(args, batched_input, model, image_embeddings, decoder_iter = False)
                
                _, _, _,src_t = prompt_and_decoder(args, batched_input, model_t, image_embeddings, decoder_iter = False)
                loss = criterion(masks, labels, iou_predictions)+1.5*criterion_gaze(masks,labels,heatmap)+0.3*ga(src_t,src,heatmap.float())
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=False)

            else:
                labels = batched_input["label"]
                heatmap = batched_input["heatmap"]
                image_embeddings = model.lora_vit.image_encoder(batched_input["image"])

                batch, _, _, _ = image_embeddings.shape
                image_embeddings_repeat = []
                for i in range(batch):
                    image_embed = image_embeddings[i]
                    image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                    image_embeddings_repeat.append(image_embed)
                image_embeddings = torch.cat(image_embeddings_repeat, dim=0)
                masks, low_res_masks, iou_predictions,src = prompt_and_decoder_lora(args, batched_input, model, image_embeddings, decoder_iter = False)
                save_image(masks.cpu(), 'tensor_image.png')
                _, _, _,src_t = prompt_and_decoder(args, batched_input, model_t, image_embeddings, decoder_iter = False)
                loss = criterion(masks, labels, iou_predictions)+0.3*gca(src_t,src,heatmap.float())+1.2*criterion_gaze(masks,labels,heatmap)
                loss.backward(retain_graph=False)

            optimizer.step()
            optimizer.zero_grad()

            if int(batch+1) % 50 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch+1}, first {flag} prompt: {SegMetrics(masks, labels, args.metrics)}')

            point_num = random.choice(args.point_list)
            #batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
            batched_input = generate_point1(masks, labels, low_res_masks, batched_input, point_num)
            batched_input = to_device(batched_input, args.device)
        
            image_embeddings = image_embeddings.detach().clone()
            for n, value in model.named_parameters():
                if "image_encoder" in n:
                    value.requires_grad = False
                else:
                    value.requires_grad = True

            init_mask_num = np.random.randint(1, args.iter_point - 1)
            for iter in range(args.iter_point):
                if iter == init_mask_num or iter == args.iter_point - 1:
                    batched_input = setting_prompt_none(batched_input)

                if args.use_amp:
                    masks, low_res_masks, iou_predictions,src = prompt_and_decoder_lora(args, batched_input, model, image_embeddings, decoder_iter=True)
                    _, _, _,src_t = prompt_and_decoder(args, batched_input, model_t, image_embeddings, decoder_iter = False)
                    loss = criterion(masks, labels, iou_predictions)+criterion_gaze(masks,labels,heatmap)
                    with amp.scale_loss(loss,  optimizer) as scaled_loss:
                        scaled_loss.backward(retain_graph=True)
                else:
                    masks, low_res_masks, iou_predictions,src = prompt_and_decoder_lora(args, batched_input, model, image_embeddings, decoder_iter=True)
                    _, _, _,src_t = prompt_and_decoder(args, batched_input, model_t, image_embeddings, decoder_iter = False)
                    loss = criterion(masks, labels, iou_predictions)+1.5*criterion_gaze(masks,labels,heatmap)
                    loss.backward(retain_graph=True)
                    
                optimizer.step()
                optimizer.zero_grad()
              
                if iter != args.iter_point - 1:
                    point_num = random.choice(args.point_list)
                    batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
                    batched_input = to_device(batched_input, args.device)
           
                if int(batch+1) % 50 == 0:
                    if iter == init_mask_num or iter == args.iter_point - 1:
                        print(f'Epoch: {epoch+1}, Batch: {batch+1}, mask prompt: {SegMetrics(masks, labels, args.metrics)}')
                    else:
                        print(f'Epoch: {epoch+1}, Batch: {batch+1}, point {point_num} prompt: { SegMetrics(masks, labels, args.metrics)}')

            if int(batch+1) % 200 == 0:
                print(f"epoch:{epoch+1}, iteration:{batch+1}, loss:{loss.item()}")
                save_path = os.path.join(f"{args.work_dir}/models", args.run_name, f"epoch{epoch+1}_batch{batch+1}_sam.pth")
                save_path_lora = os.path.join(f"{args.work_dir}/models", args.run_name, f"epoch{epoch+1}_batch{batch+1}_sam_lora.pth")
                state = {'model': model.state_dict(), 'optimizer': optimizer}
                model.save_lora_parameters(save_path_lora)
                torch.save(state, save_path)

            train_losses.append(loss.item())

            gpu_info = {}
            gpu_info['gpu_name'] = args.device 
            train_loader.set_postfix(train_loss=loss.item(), gpu_info=gpu_info)

            train_batch_metrics = SegMetrics(masks, labels, args.metrics)
            train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]

    return train_losses, train_iter_metrics



def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    sam = sam_model_registry[args.model_type](args).to(args.device)
    model= LoRA_Sam(sam,r = 4).to(args.device)
    model_t = sam_model_registry[args.model_type](args).to(args.device)
    ga=GALoss().to(args.device)
    params_list = nn.ModuleList([])
    params_list.append(model)
    params_list.append(ga)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(params_list.parameters(), lr=args.lr)
    criterion = FocalDiceloss_IoULoss()
    criterion_gaze = GazeWeightedCrossEntropyLoss()

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma = 0.5)
        print('*******Use MultiStepLR')
    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f)
            model.load_state_dict(checkpoint['model'])
            model_t.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            print(f"*******load {args.resume}")

    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        print("*******Mixed precision with Apex")
    else:
        print('*******Do not use mixed precision')

    train_dataset = TrainingDataset(args.data_path, image_size=args.image_size, mode='train', point_num=3, mask_num=args.mask_num, requires_name = False)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=8)
    print('*******Train data:', len(train_dataset))   

    loggers = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))

    best_loss = 1e10
    l = len(train_loader)

    for epoch in range(0, args.epochs):
        model.train()
        model_t.train()
        train_metrics = {}
        start = time.time()
        os.makedirs(os.path.join(f"{args.work_dir}/models", args.run_name), exist_ok=True)
        train_losses, train_iter_metrics = train_one_epoch(args, model,model_t, optimizer, train_loader, epoch, criterion,criterion_gaze,ga)

        if args.lr_scheduler is not None:
            scheduler.step()

        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}

        average_loss = np.mean(train_losses)
        lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr
        loggers.info(f"epoch: {epoch + 1}, lr: {lr}, Train loss: {average_loss:.4f}, metrics: {train_metrics}")

        if average_loss < best_loss:
            best_loss = average_loss
            save_path = os.path.join(args.work_dir, "models", args.run_name, f"epoch{epoch+1}_sam.pth")
            save_path_lora = os.path.join(args.work_dir, "models", args.run_name, f"epoch{epoch+1}_sam_lora.safetensors")
            state = {'model': model.float().state_dict(), 'optimizer': optimizer}
            torch.save(state, save_path)
            #model.save_lora_parameters(save_path_lora)
            if args.use_amp:
                model = model.half()

        end = time.time()
        print("Run epoch time: %.2fs" % (end - start))


if __name__ == '__main__':
    args = parse_args()
    main(args)


