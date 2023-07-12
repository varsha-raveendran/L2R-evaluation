#evaluation.py
import json
import nibabel as nib
import os.path
from pathlib import Path
import pandas
import argparse

from utils import *
from surface_distance import *
import torch.nn.functional as F
import os
import torch
import wandb
from wandb_utils import *
from PIL import Image

def image_norm(img):
    max_v = np.max(img)
    min_v = np.min(img)

    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img

def evaluate_L2R(INPUT_PATH,GT_PATH,OUTPUT_PATH,JSON_PATH,verbose=False, output_file_path="Results", slice_no=96):
    columns=["fix_name", "mov_name", "fix_img", "mov_img", "warp_img", "flow", "ncc"]
    wandb_img_table = create_wandb_table(columns)
    
    columns=["fix_name", "mov_name", "fix_mask", "mov_mask", "warp_mask", "dice", "hd95"]
    wandb_mask_table = create_wandb_table(columns)
    
    columns=["fix_name", "mov_name", "fix_kp", "mov_kp", "warp_kp", "tre"]
    wandb_kp_table = create_wandb_table(columns)
    
    columns=["fix_name", "mov_name", "dice", "hd95", "tre", "ncc"]
    wandb_metrics_table = create_wandb_table(columns)   
    
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    name=data['task_name']
    expected_shape=np.array(data['expected_shape'])
    evaluation_methods_metrics=[tmp['metric'] for tmp in data['evaluation_methods']]
    if 'masked_evaluation' in data:
        use_mask = data['masked_evaluation']
    else:
        use_mask = False
    eval_pairs=data['eval_pairs']
    len_eval_pairs=len(eval_pairs)

    
    #Check if files are complete beforehand
    for idx, pair in enumerate(eval_pairs):
        disp_name='flow_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii.gz')
        disp_path=os.path.join(INPUT_PATH, disp_name)
        # if os.path.isfile(disp_path):
        #     continue
        # else:
        #     raise_missing_file_error(disp_name)

    #Dataframe for Case results
    cases_results=pandas.DataFrame()
    cases_details={}

    img_results = []
    mask_results = []
    metrics_results = []
    kp_results = []
        
    lncc_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(
        spatial_dims=3, kernel_size=3, kernel_type="rectangular", reduction="mean"
    )
    
    if verbose:
        print(f"Evaluate {len_eval_pairs} cases for: {[tmp['name'] for tmp in data['evaluation_methods']]}")
    if use_mask and verbose:
        print("Will use masks for evaluation.")
    for idx, pair in enumerate(eval_pairs):
        case_results={}
        case_details={}
        
        pair_img_results = {}
        pair_mask_results = {}
        pair_metrics_results = {}
        pair_kp_results = {}

        fix_name=pair['fixed'][-16:-12]
        mov_name=pair['moving'][-16:-12]
        
        pair_img_results["fix_name"] = fix_name
        pair_img_results["mov_name"] = mov_name
        pair_mask_results["fix_name"] = fix_name
        pair_mask_results["mov_name"] = mov_name
        pair_metrics_results["fix_name"] = fix_name
        pair_metrics_results["mov_name"] = mov_name
        pair_kp_results["fix_name"] = fix_name
        pair_kp_results["mov_name"] = mov_name
        fix_img_path=os.path.join(GT_PATH, pair['fixed'])
        mov_img_path=os.path.join(GT_PATH, pair['moving'])
        
        fix_label_path=os.path.join(GT_PATH, pair['fixed'].replace('images','masks'))
        mov_label_path=os.path.join(GT_PATH, pair['moving'].replace('images','masks'))
        
        warped_img = None
        fixed_img = None
        moving_img = None
        
        #with nii.gz
        disp_path=os.path.join(INPUT_PATH, 'flow_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii.gz'))
        disp_field=nib.load(disp_path).get_fdata()
        # disp_field=nib.load(disp_path).get_fdata()
        print(disp_field.shape)
        
        grid_disp_path = os.path.join(output_file_path + '/disp_fields', f'flow_{str(fix_name).zfill(4)}_{str(mov_name).zfill(4)}.png')
        preview_3D_deformation( torch.from_numpy(disp_field).permute(3,0,1,2), 5, linewidth=1, color="darkblue", file_path=grid_disp_path)
        pair_img_results["flow"] = grid_disp_path
        
        shape = np.array(disp_field.shape)
        # if not np.all(shape==expected_shape):
        #     raise_shape_error(disp_name, shape, expected_shape)

        ## load neccessary files 
        if any([True for eval_ in ['tre'] if eval_ in evaluation_methods_metrics]):
            spacing_fix=nib.load(os.path.join(GT_PATH, pair['fixed'])).header.get_zooms()[:3]
            spacing_mov=nib.load(os.path.join(GT_PATH, pair['moving'])).header.get_zooms()[:3]
            
        if any([True for eval_ in ['ncc'] if eval_ in evaluation_methods_metrics]):
            fixed_img=image_norm(nib.load(fix_img_path).get_fdata())
            moving_img=image_norm(nib.load(mov_img_path).get_fdata())
            
            
            D,H,W = fixed_img.shape
            
            # fixed_img=F.interpolate(torch.from_numpy(fixed_img).view(1,1,D,H,W),size=(D//2,H//2,W//2),mode='trilinear').squeeze().numpy()
            # moving_img=F.interpolate(torch.from_numpy(moving_img).view(1,1,D,H,W),size=(D//2,H//2,W//2),mode='trilinear').squeeze().numpy()
            # D,H,W = fixed_img.shape
            identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
            
            warped_img = map_coordinates(moving_img, identity + disp_field.transpose(3,0,1,2), order=0)
            # im = Image.fromarray(warped_img[:,H//2,:].astype(np.uint8))
            moved_path = os.path.join(output_file_path + '/moved_imgs', f'moved_{str(fix_name).zfill(4)}_{str(mov_name).zfill(4)}.png')
            fixed_path = os.path.join(output_file_path + '/moved_imgs', f'fixed_{str(fix_name).zfill(4)}.png')
            moving_path = os.path.join(output_file_path + '/moved_imgs', f'moving_{str(fix_name).zfill(4)}.png')
            # im.save(moved_path)
            pair_img_results["fix_img"] = fixed_path
            pair_img_results["mov_img"] = moving_path
            pair_img_results["warp_img"] = moved_path
            print(warped_img.shape)
            plot_img(warped_img[:,slice_no,:], moved_path, cmap="gray")
            plot_img(fixed_img[:,slice_no,:], fixed_path, cmap="gray")
            plot_img(moving_img[:,slice_no,:], moving_path, cmap="gray")
            
            moved_path = os.path.join(output_file_path + '/moved_imgs', f'moved_{str(fix_name).zfill(4)}_{str(mov_name).zfill(4)}_3views.png')
            fixed_path = os.path.join(output_file_path + '/moved_imgs', f'fixed_{str(fix_name).zfill(4)}_3views.png')
            moving_path = os.path.join(output_file_path + '/moved_imgs', f'moving_{str(fix_name).zfill(4)}_3views.png')
            preview_image(warped_img, normalize_by="slice", cmap="gray", file_path=moved_path)
            preview_image(fixed_img, normalize_by="slice", cmap="gray", file_path=fixed_path)
            preview_image(moving_img, normalize_by="slice", cmap="gray", file_path=moving_path)  

        if any([True for eval_ in ['dice','hd95'] if eval_ in evaluation_methods_metrics]):
            fixed_seg=nib.load(fix_label_path).get_fdata()
            moving_seg=nib.load(mov_label_path).get_fdata()
            
            D,H,W = fixed_seg.shape
            # fixed_seg=F.interpolate(torch.from_numpy(fixed_seg).view(1,1,D,H,W),size=(D//2,H//2,W//2),mode='nearest').squeeze().numpy()
            # moving_seg=F.interpolate(torch.from_numpy(moving_seg).view(1,1,D,H,W),size=(D//2,H//2,W//2),mode='nearest').squeeze().numpy()
            # D,H,W = fixed_seg.shape
            print(moving_seg.shape)
            identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
           
            warped_seg = map_coordinates(moving_seg, identity + disp_field.transpose(3,0,1,2), order=0)
            print(warped_seg.shape)
            seg_path = os.path.join(output_file_path + '/moved_masks', f'moved_{str(fix_name).zfill(4)}_{str(mov_name).zfill(4)}_3views.png')
            fixed_path = os.path.join(output_file_path + '/moved_masks', f'fixed_{str(fix_name).zfill(4)}_3views.png')
            moving_path = os.path.join(output_file_path + '/moved_masks', f'moving_{str(fix_name).zfill(4)}_3views.png')
            preview_image(warped_seg, normalize_by="slice",  file_path=seg_path)
            preview_image(fixed_seg, normalize_by="slice",  file_path=fixed_path)
            preview_image(moving_seg, normalize_by="slice",  file_path=moving_path)
        
        
            seg_path = os.path.join(output_file_path + '/moved_masks', f'moved_{str(fix_name).zfill(4)}_{str(mov_name).zfill(4)}.png')
            fixed_path = os.path.join(output_file_path + '/moved_masks', f'fixed_{str(fix_name).zfill(4)}.png')
            moving_path = os.path.join(output_file_path + '/moved_masks', f'moving_{str(fix_name).zfill(4)}.png')
            # im.save(seg_path)
            plot_img(warped_seg[:,slice_no,:], seg_path)
            plot_img(fixed_seg[:,slice_no,:], fixed_path)
            plot_img(moving_seg[:,slice_no,:], moving_path)
            pair_mask_results["fix_mask"] = fixed_path
            pair_mask_results["mov_mask"] = moving_path
            pair_mask_results["warp_mask"] = seg_path
            
        
        if use_mask:
            mask_path= os.path.join(GT_PATH, pair['fixed'].replace('images','masks'))
            if os.path.exists(mask_path):
                mask=nib.load(mask_path).get_fdata()
                mask_ready=True
            else:
                print(f'Tried to use mask but did not find {mask_path}. Will evaluate without mask.')
                mask_ready=False

                
        ## iterate over designated evaluation metrics
        for _eval in data['evaluation_methods']:
            _name=_eval['name']

            if 'ncc' == _eval['metric']:
                ncc =  lncc_loss(torch.from_numpy(warped_img).unsqueeze(0).unsqueeze(0), torch.from_numpy(fixed_img).unsqueeze(0).unsqueeze(0)).item()
                pair_img_results["ncc"] = ncc
                pair_metrics_results["ncc"] = ncc
                case_results[_name]= ncc
                
            ### SDlogJ
            
            if 'sdlogj' == _eval['metric']:
                # jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :].transpose((0,4,1,2,3))) + 3).clip(0.000000001, 1000000000)
                # jac_det = jacobian_determinant_(disp_field)
                jac_det = jacobian_determinant_vxm(disp_field)
                log_jac_det = np.log(abs(jac_det))
                if use_mask and mask_ready:
                    case_results[_name]=np.ma.MaskedArray(log_jac_det, 1-mask[2:-2, 2:-2, 2:-2]).std()
                else:
                    case_results[_name]=log_jac_det.std()
                case_results['num_foldings']=(jac_det <= 0).astype(float).sum()
                #copy into detailed evaluation
                case_details[_name]=case_results[_name]
                case_details['num_foldings']=case_results['num_foldings']

            ### DSC
            if 'dice' == _eval['metric']:
                labels = _eval['labels']
                dice = compute_dice(fixed_seg,moving_seg,warped_seg,labels)
                case_results[_name],case_details[_name]=dice
                
                pair_mask_results["dice"] = dice[0]
                pair_metrics_results["dice"] = dice[0]
                
                fix_one_hot = monai.networks.utils.one_hot(torch.from_numpy(fixed_seg).unsqueeze(0), num_classes=2,dim=0 )
                warp_one_hot = monai.networks.utils.one_hot(torch.from_numpy(warped_seg).unsqueeze(0), num_classes=2,dim=0 )
                dice_monai = monai.metrics.compute_dice(warp_one_hot.unsqueeze(0),fix_one_hot.unsqueeze(0),include_background=False)
                print("dice_monai", dice_monai[0])
                

            ### HD95
            if 'hd95' == _eval['metric']:
                labels = _eval['labels']
                hd95 = compute_hd95(fixed_seg,moving_seg,warped_seg,labels)
                case_results[_name],case_details[_name]=hd95
                pair_mask_results["hd95"] = hd95[0]
                pair_metrics_results["hd95"] = hd95[0]
                fix_one_hot = monai.networks.utils.one_hot(torch.from_numpy(fixed_seg).unsqueeze(0), num_classes=2,dim=0 )
                warp_one_hot = monai.networks.utils.one_hot(torch.from_numpy(warped_seg).unsqueeze(0), num_classes=2,dim=0 )
                hd = monai.metrics.hausdorff_distance.compute_hausdorff_distance(warp_one_hot.unsqueeze(0),fix_one_hot.unsqueeze(0), percentile=95)
                print("hd", hd.mean())
        
            ### TRE
            if 'tre' == _eval['metric']:
                destination = _eval['dest']
                lms_fix_path = os.path.join(GT_PATH, pair['fixed'].replace('images', destination).replace('.nii.gz','.csv'))
                lms_mov_path = os.path.join(GT_PATH, pair['moving'].replace('images', destination).replace('.nii.gz','.csv'))
                fix_lms = np.loadtxt(lms_fix_path, delimiter=',') 
                mov_lms = np.loadtxt(lms_mov_path, delimiter=',') 
                print("mov spacing" , spacing_mov)
                tre, fix_lms_warped = compute_tre(fix_lms, mov_lms, disp_field ,spacing_fix, spacing_mov)
                case_results[_name]=tre.mean()
                
                fix_path = os.path.join(output_file_path + '/moved_kps', f'fixed_{str(fix_name).zfill(4)}.png')
                mov_path = os.path.join(output_file_path + '/moved_kps', f'moving_{str(mov_name).zfill(4)}.png')
                
                plot_keypoints(fix_lms,fixed_img,slice_no , fix_path)
                plot_keypoints(mov_lms,moving_img,slice_no , mov_path)
                
                pair_kp_results["fix_kp"] = fix_path
                pair_kp_results["mov_kp"] = mov_path
                
                warp_path = os.path.join(output_file_path + '/moved_kps', f'moved_{str(fix_name).zfill(4)}_{str(mov_name).zfill(4)}.png')
                plot_keypoints(fix_lms_warped,warped_img,slice_no , warp_path)                
                pair_kp_results["warp_kp"] = warp_path
                pair_kp_results["tre"] = tre.mean()
                pair_metrics_results["tre"] =  tre.mean()

        img_results.append(pair_img_results)
        mask_results.append(pair_mask_results)
        metrics_results.append(pair_metrics_results)
        kp_results.append(pair_kp_results)  
        
        if verbose:
            print(f'case_results [{idx}]: {case_results}')
        cases_results=pandas.concat([cases_results, pandas.DataFrame(case_results, index=[0])], ignore_index=True)
        cases_details[idx]=case_details
             
        
    aggregated_results = {}   
    for col in cases_results.columns:
        aggregated_results[col] = {'30': cases_results[col].quantile(.3),
                                  'std': cases_results[col].std(),
                                  'mean': cases_results[col].mean()}
    final_results={
        name: {
            "case": cases_results.to_dict(),
            "aggregates": aggregated_results,
            "detailed" : cases_details
    }}

    #print(f'aggregated_results [{name}]: {aggregated_results}')
    if verbose:
        print(json.dumps(aggregated_results, indent=4))
    
    with open(os.path.join(OUTPUT_PATH), 'w') as f:
        json.dump(final_results, f, indent=4)
        
    log_wandb_table_img(img_results, wandb_img_table, "Image Results")
    log_wandb_table_mask(mask_results, wandb_mask_table, "Mask Results")
    log_wandb_table_metrics(metrics_results, wandb_metrics_table, "Metrics Results")
    log_wandb_table_kp(kp_results, wandb_kp_table, "Keypoints Results")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description='L2R Evaluation script\n'\
    'Docker PATHS:\n'\
    'DEFAULT_INPUT_PATH = Path("/input/")\n'\
    'DEFAULT_GROUND_TRUTH_PATH = Path("/opt/evaluation/ground-truth/")\n'\
    'DEFAULT_EVALUATION_OUTPUT_FILE_PATH = Path("/output/metrics.json")')
    parser.add_argument("-i","--input", dest="input_path", help="path to deformation_field", default="test")
    parser.add_argument("-d","--data", dest="gt_path", help="path to data", default="ground-truth")
    parser.add_argument("-o","--output", dest="output_path", help="path to write results(e.g. 'results/metrics.json')", default="metrics.json")
    parser.add_argument("-c","--config", dest="config_path", help="path to config json-File (e.g. 'evaluation_config.json')", default='ground-truth/evaluation_config.json') 
    parser.add_argument("-v","--verbose", dest="verbose", action='store_true', default=False)
    
    parser.add_argument("-m","--model_path", dest="model_path", help="path to model")
    parser.add_argument("-of","--output_files", dest="output_files", help="path to write image outputs", default="Results")
    parser.add_argument("-s","--slice_no", dest="slice_no", help="slice nuber to save/plot", default=96)
    parser.add_argument("-p","--project_name", dest="project_name", help="project_name")
    args= parser.parse_args()
    
    run = wandb.init(project=args.project_name, job_type='evaluation' )
    
    wandb.config.update(args)
    log_model_wandb(args.model_path)
    os.makedirs(args.output_files, exist_ok=True)
    os.makedirs(args.output_files + "/" + "moved_imgs", exist_ok=True)
    os.makedirs(args.output_files + "/" + "moved_masks", exist_ok=True)
    os.makedirs(args.output_files + "/" + "moved_kps", exist_ok=True)
    os.makedirs(args.output_files + "/" + "disp_fields", exist_ok=True)
    
    evaluate_L2R(args.input_path, args.gt_path, args.output_path, args.config_path, args.verbose, args.output_files, int(args.slice_no))
    
