#https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2.git 깃허브 Ultra-Fast-Lane-Detection-V2 demo.py 코드 사용

import torch, os, cv2
from utils.dist_utils import dist_print
import torch, os
from utils.common import merge_config, get_model
import tqdm
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset

def pred2coords(pred, row_anchor, col_anchor, local_width = 1, original_image_width = 1640, original_image_height = 590):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_row = pred['exist_row'].argmax(1).cpu()
    # n, num_cls, num_lanes

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_col = pred['exist_col'].argmax(1).cpu()
    # n, num_cls, num_lanes

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []

    row_lane_idx = [1,2]
    col_lane_idx = [0,3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0,:,i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_row'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0,:,i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_col'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5

                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

    return coords

def process_warning_and_draw(vis, coords, img_w, img_h):
    # 1. [해결안 A] 카메라 장착 위치 오프셋 보정
    # 차선 중앙을 잘 달리는데도 자꾸 위험이 뜬다면, 화면에 나오는 Offset 수치를 보고 
    # 이 값을 + 또는 - (예: 15, -20 등)로 입력하여 0 근처가 되도록 캘리브레이션 하세요.
    camera_offset = 0  
    X_car = (img_w // 2) + camera_offset
    
    # 2. [해결안 B] 판단 임계값 완화 (기존 4%/8% -> 7%/13%로 상향 조정)
    # 1640 해상도 기준 경고는 약 114픽셀, 위험은 약 213픽셀 이상 벗어나야 작동합니다.
    theta_warning = int(img_w * 0.07)  
    theta_danger = int(img_w * 0.13)   
    
    COLOR_NORMAL = (255, 120, 0)    # 정상: 푸른색
    COLOR_WARNING = (0, 165, 255)   # 경고: 주황색
    COLOR_DANGER = (0, 0, 255)      # 위험: 빨간색
    COLOR_LOST = (0, 255, 255)      # 미검출: 노란색
    
    current_color = COLOR_NORMAL
    status_text = "STATUS: NORMAL"
    
    lane1 = coords[0] if len(coords) > 0 else []
    lane2 = coords[1] if len(coords) > 1 else []
    
    if len(lane1) > 0 and len(lane2) > 0:
        # 3. [해결안 C] 단 하나의 점([-1])만 쓰지 않고, 하단 5개 점의 평균을 내어 지터링 노이즈 제거
        sample_num = min(5, len(lane1), len(lane2)) 
        
        X1_bottoms = [p[0] for p in lane1[-sample_num:]]
        X2_bottoms = [p[0] for p in lane2[-sample_num:]]
        
        X1_avg = sum(X1_bottoms) // len(X1_bottoms)
        X2_avg = sum(X2_bottoms) // len(X2_bottoms)
        
        X_lane = (X1_avg + X2_avg) // 2
        D_offset = abs(X_car - X_lane)
        
        if D_offset > theta_danger:
            current_color = COLOR_DANGER
            status_text = "WARNING: LANE DEPARTURE DANGER!"
        elif D_offset > theta_warning:
            current_color = COLOR_WARNING
            status_text = "WARNING: LATERALLY DRIFTING"
            
        # 4. [디버깅 툴] 현재 계산되는 오차 픽셀(px) 값을 화면 우측 상단에 실시간으로 출력합니다.
        # 이 숫자가 평소에 얼마까지 튀는지 보고 위의 theta_warning, theta_danger를 튜닝하면 편리합니다.
        cv2.putText(vis, f"Offset: {D_offset}px", (img_w - 220, 42), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
    else:
        current_color = COLOR_LOST
        status_text = "STATUS: LANE UNSTABLE / LOST"
        X_lane = X_car # 차선 유실 시 게이지 바는 중앙 유지
        
    # 차선 포인트 그리기
    for lane in coords:
        for coord in lane:
            cv2.circle(vis, coord, 5, current_color, -1)
            
    # 상단 알림창 UI
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (img_w, 60), current_color, -1)
    cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
    
    cv2.putText(vis, status_text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
                
    # 하단 대시보드 게이지 바
    cv2.line(vis, (X_car, img_h - 20), (X_lane, img_h - 20), current_color, 6)
    cv2.circle(vis, (X_car, img_h - 20), 8, (0, 255, 0), -1) 
        
    return vis

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()
    cfg.batch_size = 1
    print('setting batch_size to 1 for demo generation')

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = get_model(cfg)

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if cfg.dataset == 'CULane':
        #생성하고 싶은 테스트 데이터 영상을 splits 리스트에 담으면 됩니다. 
        #splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
        splits = ['test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, 'list/test_split/'+split),img_transform = img_transforms, crop_size = cfg.train_height) for split in splits]
        img_w, img_h = 1640, 590
    elif cfg.dataset == 'Tusimple':
        splits = ['test.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, split),img_transform = img_transforms, crop_size = cfg.train_height) for split in splits]
        img_w, img_h = 1280, 720
    else:
        raise NotImplementedError
    for split, dataset in zip(splits, datasets):
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print(split[:-3]+'avi')
        vout = cv2.VideoWriter(split[:-3]+'avi', fourcc , 30.0, (img_w, img_h))
        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            imgs = imgs.cuda()
            with torch.no_grad():
                pred = net(imgs)

            vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
            coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width = img_w, original_image_height = img_h)
            '''
            for lane in coords:
                for coord in lane:
                    cv2.circle(vis,coord,5,(0,255,0),-1)
            '''
            vis = process_warning_and_draw(vis, coords, img_w, img_h)
            vout.write(vis)
        
        vout.release()
