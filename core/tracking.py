import cv2
import numpy as np
import math

def get_lines(frameWidth, frameHeight):
    # Create coordinates for 2 ROI lines
    global line1_pt1, line1_pt2, line2_pt1, line2_pt2
    line1_pt1 = (frameWidth - 120, frameHeight - 30)
    line1_pt2 = (frameWidth + 180, frameHeight - 30)

    line2_pt1 = (frameWidth - 120, frameHeight + 30)
    line2_pt2 = (frameWidth + 180, frameHeight + 30)

    return (line1_pt1,line1_pt2), (line2_pt1,line2_pt2)

def track_and_count(frame, tracking_objects, track_Id, center_pts_cur_frame, for_line, bottle, fallen_bottle):

    tracking_objects_copy = tracking_objects.copy()
    center_pts_cur_frame_copy = center_pts_cur_frame.copy()

    for object_id, pt2 in tracking_objects_copy.items():
        object_exists = False
        for pt in center_pts_cur_frame_copy:
            distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

            if distance < 32:
                tracking_objects[object_id] = pt
                object_exists = True
                if pt in center_pts_cur_frame:
                    # print('Yes')
                    center_pts_cur_frame.remove(pt)
                continue
        
        if not object_exists:
            tracking_objects.pop(object_id)
            # for_line.remove(object_id)
    
    for pt in center_pts_cur_frame:
        tracking_objects[track_Id] = pt
        track_Id += 1
    

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt[:2], 2, (211,211,211), -1)
        # cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,0,255), 1)
        # See if bottle passes 2nd line from above
        # Increment the count if object_id unique
        if object_id not in for_line:
            # Using Cross Product
            dist = ((line2_pt2[0] - line2_pt1[0])*(pt[1] - line2_pt1[1]) - (line2_pt2[1] - line2_pt1[1])*(pt[0] - line2_pt1[0])) // 100
            if dist < 0 and dist > -40:
                # Check the class and increment accordingly
                if pt[2] == 0:
                    bottle += 1
                else:
                    fallen_bottle += 1 
                for_line.append(object_id)
        
        # See if bottle passes 1nd line from above
        if object_id not in for_line:
            dist2 = ((line1_pt2[0] - line1_pt1[0])*(pt[1] - line1_pt1[1]) - (line1_pt2[1] - line1_pt1[1])*(pt[0] - line1_pt1[0])) // 100
            if dist2 < 0 and dist2 > -40:
                if pt[2] == 0:
                    bottle += 1
                else:
                    fallen_bottle += 1 

    return tracking_objects, track_Id, center_pts_cur_frame, for_line, bottle, fallen_bottle

def get_fps(new_frame_time, prev_frame_time):
    # Get FPS
    fps = 1/(new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    return fps, prev_frame_time