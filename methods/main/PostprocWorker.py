import numpy as np
import cv2
from math import ceil

class PostprocWorker:
    @staticmethod
    def first_wave_postprocessing(model_results, nodata, bounds, id_counter, id_counter_increment):
        fields = []
        masks = []
        bbox = []
        ids = []

        lbound = bounds
        clip_mask = nodata[0, :, :]
        filter_mask = nodata[1, :, :]

        for result in model_results:
            if result.masks is None or len(result.masks) == 0:
                continue

            dnp = result.masks.data.cpu().numpy().astype("uint8")
            for j in range(dnp.shape[0]):
                l_bbox = result.boxes[j].xyxy.cpu()

                min_x, min_y, max_x, max_y = (int(l_bbox[0][0]), int(l_bbox[0][1]), int(ceil(l_bbox[0][2])), int(ceil(l_bbox[0][3])))
                min_x = max(0, min_x - 4)
                min_y = max(0, min_y - 4)
                max_x = min(512, max_x + 4)
                max_y = min(512, max_y + 4)

                fields.append(dnp[j][min_y:max_y, min_x:max_x].copy())
                masks.append([
                    clip_mask[min_y:max_y, min_x:max_x].copy(),
                    filter_mask[min_y:max_y, min_x:max_x].copy()
                ])
                bbox.append([min_x, max_x, min_y, max_y])

                ids.append(id_counter)
                id_counter += id_counter_increment

        return {
            "fields": fields,
            "masks": masks,
            "bboxes": bbox,
            "ids": ids,
            "bounds": lbound
        }

    @staticmethod
    def process_fields(entry, mapping_dict, area_dict, instances_info, weight_info, config):
        dst_instances = np.ndarray(instances_info[1], dtype="int32", buffer=instances_info[0].buf)
        dst_weigths = np.ndarray(weight_info[1], dtype="float32", buffer=weight_info[0].buf)

        MIN_AREA = config["pixel_area_threshold"]
        MIN_REL_AREA = config["remaining_area_threshold"]
        COMPOSE_MERGE_IOU = config["compose_merge_iou"]

        MERGING_EDGE_WIDTH = config["merging_edge_width"]

        MERGE_IOU = config["merge_iou"]
        MERGE_EDGE_IOU = config["merge_edge_iou"]
        MERGE_EDGE_PIXELS = config["merge_edge_pixels"]

        MERGE_RELATIVE_AREA_THRESHOLD = config["merge_relative_area_threshold"]
        MERGE_ASYMETRIC_MERGING_PIXEL_AREA_THRESHOLD = config["merge_asymetric_pixel_area_threshold"]
        MERGE_ASYMETRYC_MERGING_RELATIVE_AREA_THRESHOLD = config["merge_asymetric_relative_area_threshold"]

        fields = entry["fields"]
        masks = entry["masks"]
        bboxes = entry["bboxes"]
        ids = entry["ids"]

        areas = []
        rel_areas = []
        # clean fields
        for j in range(len(fields)):
            min_x, max_x, min_y, max_y = bboxes[j]
            clip_mask, filter_mask = masks[j]
            field = (PostprocWorker.get_biggest_component_raster(fields[j]) > 0) * (clip_mask > 0)
            fields[j] = field

            iniatial_area = np.sum(field)
            valid_area = np.sum(field * (filter_mask > 0))

            areas.append(iniatial_area)

            if iniatial_area == 0:
                rel_areas.append(0)
            else:
                rel_areas.append(valid_area / iniatial_area)
        # end clean

        order_b2s = np.argsort(np.array(areas))[::-1]
        instances = np.zeros((512, 512), dtype="int32")
        weights = np.zeros((512, 512), dtype="float32")

        write_id = ids.copy()

        id_area = { ids[i]: areas[i] for i in range(len(fields)) }
        # compose fields
        for index in order_b2s:
            area, rel_area = areas[index], rel_areas[index]
            if area < MIN_AREA or rel_area < MIN_REL_AREA:
                write_id[index] = 0
                continue

            index_id = ids[index]
            min_x, max_x, min_y, max_y = bboxes[index]

            field = fields[index]
            npu_intersections = np.unique(instances[min_y:max_y, min_x:max_x][field], return_counts=True)
            intersection_areas_dict = dict(zip(npu_intersections[0], npu_intersections[1]))

            for key_id in intersection_areas_dict.keys():
                if key_id == 0:
                    continue

                area_inter = intersection_areas_dict[key_id]
                area_key = id_area[key_id]

                if area_key < MIN_AREA:
                    continue
                
                iou = area_inter / max(1, float(area + area_key - area_inter))

                # major intersection of fields -> replace current field with id of one with the intersection
                if iou > COMPOSE_MERGE_IOU:
                    id_area[key_id] += area - area_inter
                    id_area[index_id] = 0

                    instances[min_y:max_y, min_x:max_x][field] = key_id
                    write_id[index] = key_id
                    break

            if id_area[index_id] >= MIN_AREA:
                instances[min_y:max_y, min_x:max_x][field] = index_id

                for key_id in intersection_areas_dict.keys():
                    if key_id == 0:
                        continue

                    area_inter = intersection_areas_dict[key_id]
                    id_area[key_id] -= area_inter

        instances[:, :] = 0
        for index in order_b2s:
            id = write_id[index]
            if id == 0:
                continue

            area = id_area[id]
            if area < MIN_AREA:
                continue
            
            min_x, max_x, min_y, max_y = bboxes[index]
            field = fields[index]

            instances[min_y:max_y, min_x:max_x][field] = id
            weights[min_y:max_y, min_x:max_x][field] = 1.0 / area

            area_dict[id] = area
            area_dict[id | 1] = area
            mapping_dict[id | 1] = [int(id)]
        # end compose

        instances[:MERGING_EDGE_WIDTH, :] |= 1
        instances[-MERGING_EDGE_WIDTH:, :] |= 1
        instances[:, :MERGING_EDGE_WIDTH] |= 1
        instances[:, -MERGING_EDGE_WIDTH:] |= 1

        inregion_bx, inregion_by, inregion_width, inregion_height = entry["bounds"]["inregion"]
        instances = cv2.resize(instances, (inregion_height, inregion_width))

        posX_begin = max(0, inregion_bx)
        posX_end = min(instances_info[1][1], inregion_bx + inregion_width)

        posY_begin = max(0, inregion_by)
        posY_end = min(instances_info[1][0], inregion_by + inregion_height)

        in_px_begin = posX_begin - inregion_bx
        in_px_end = posX_end - inregion_bx
        in_py_begin = posY_begin - inregion_by
        in_py_end = posY_end - inregion_by

        if posX_begin >= posX_end or posY_begin > posY_end:
            print(f"WARNING. Innatural ranges: X {posX_begin} {posX_end} Y {posY_begin} {posY_end}. Tile skipped")
            return

        old_instances = dst_instances[posY_begin:posY_end, posX_begin:posX_end]
        old_weights = dst_weigths[posY_begin:posY_end, posX_begin:posX_end]

        # # merge fields
        PostprocWorker.find_edge_mapping(old_instances, instances[in_py_begin:in_py_end, in_px_begin:in_px_end], mapping_dict, area_dict, 
                        MERGE_IOU, MERGE_EDGE_IOU, MERGE_EDGE_PIXELS,
                        MERGE_RELATIVE_AREA_THRESHOLD, MERGE_ASYMETRIC_MERGING_PIXEL_AREA_THRESHOLD, MERGE_ASYMETRYC_MERGING_RELATIVE_AREA_THRESHOLD)
        # # end merge

        write_mask = weights[in_py_begin:in_py_end, in_px_begin:in_px_end] > old_weights

        old_instances[write_mask] = instances[in_py_begin:in_py_end, in_px_begin:in_px_end][write_mask]
        old_weights[write_mask] = weights[in_py_begin:in_py_end, in_px_begin:in_px_end][write_mask]

    @staticmethod
    def find_edge_mapping(current, new, dst, area_dict, merge_iou, 
                          merge_edge_iou, merge_edge_pixels, 
                          merge_relative_area_threshold, 
                          merge_asymetric_pixel_area_threshold, 
                          merge_asymetric_relative_area_threshold):
        uniq_intersect = np.unique((current.astype("uint64") << 32) | new.astype("uint64"), return_counts=True)
        uniq_dict = dict(zip(uniq_intersect[0], uniq_intersect[1]))

        uniq_current = np.unique(current, return_counts=True)
        uniq_current_dict = dict(zip(uniq_current[0], uniq_current[1]))

        uniq_new = np.unique(new, return_counts=True)
        uniq_new_dict = dict(zip(uniq_new[0], uniq_new[1]))

        if len(uniq_dict.keys()) == 0:
            return {}

        for i in range(uniq_intersect[0].shape[0]):
            key = np.uint64(uniq_intersect[0][i])
            key_current = int(key >> np.uint64(32))
            key_new = int(key & np.uint64(0xFFFFFFFF))

            if key_current < 2 or key_new < 2:
                continue

            kc_0 = np.uint64(2 * (key_current // 2))
            kc_1 = kc_0 | np.uint64(1)
            kn_0 = np.uint64(2 * (key_new // 2))
            kn_1 = kn_0 | np.uint64(1)

            keys = [
                (kc_0 << np.uint64(32)) | kn_0,
                (kc_0 << np.uint64(32)) | kn_1,
                (kc_1 << np.uint64(32)) | kn_0,
                (kc_1 << np.uint64(32)) | kn_1,
            ]

            area_intersect = 0
            for key in keys:
                if key in uniq_dict:
                    area_intersect += uniq_dict[key]

            area_current = area_dict[int(key_current)]
            area_new = area_dict[int(key_new)]
            iou = area_intersect / max(1, area_current + area_new - area_intersect)

            case_0_is_edge_merging = (key_current % 2 == 1 or key_new % 2 == 1) and (iou > merge_edge_iou or area_intersect > merge_edge_pixels)
            case_0_is_iou_merging = iou > merge_iou

            case_0 = case_0_is_edge_merging or case_0_is_iou_merging


            area_current_local = (uniq_current_dict[kc_0] if kc_0 in uniq_current_dict else 0) + (uniq_current_dict[kc_1] if kc_1 in uniq_current_dict else 0)
            area_new_local = (uniq_new_dict[kn_0] if kn_0 in uniq_new_dict else 0) + (uniq_new_dict[kn_1] if kn_1 in uniq_new_dict else 0)


            rel_area_current = area_intersect / area_current_local
            rel_area_new = area_intersect / area_new_local

            case_1_relative_area_merging = rel_area_current > merge_relative_area_threshold and rel_area_new > merge_relative_area_threshold
            case_1_asymetrics_current = area_intersect > merge_asymetric_pixel_area_threshold and rel_area_current > merge_asymetric_relative_area_threshold
            case_1_asymetrics_new = area_intersect > merge_asymetric_pixel_area_threshold and rel_area_new > merge_asymetric_relative_area_threshold

            case_1 = case_1_relative_area_merging or case_1_asymetrics_current or case_1_asymetrics_new

            if case_0 and case_1:
                dst[int(key_new)] = dst.get(int(key_new), []) + [int(key_current)]

    @staticmethod
    def get_biggest_component_raster(mask):
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8")
        buff = mask.astype("uint8")

        if not np.any(buff):
            return buff

        buff = cv2.morphologyEx(buff, cv2.MORPH_ERODE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

        if not np.any(buff):
            return buff

        contours, _ = cv2.findContours(buff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c.squeeze(1) for c in contours]
        areas = np.array([cv2.contourArea(c) for c in contours])

        buff[:, :] = 0
        if len(areas) == 0:
            return buff

        max_contour = contours[np.argmax(areas)]
        buff = cv2.fillPoly(buff, [max_contour.astype("int32")], 1, cv2.LINE_4)
        buff = cv2.morphologyEx(buff, cv2.MORPH_DILATE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        return buff