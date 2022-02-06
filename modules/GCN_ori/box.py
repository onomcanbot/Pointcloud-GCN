import numpy as np

median_object_size_map = {
    'Cyclist': (1.76, 1.75, 0.6),
    'Van': (4.98, 2.13, 1.88),
    'Tram': (14.66, 3.61, 2.6),
    'Car': (3.88, 1.5, 1.63),
    'Misc': (2.52, 1.65, 1.51),
    'Pedestrian': (0.88, 1.77, 0.65),
    'Truck': (10.81, 3.34, 2.63),
    'Person_sitting': (0.75, 1.26, 0.59),
    # 'DontCare': (-1.0, -1.0, -1.0)
}

def box_encoding(cls_labels, points_xyz, boxes_3d,
    label_map):
    encoded_boxes_3d = np.copy(boxes_3d)
    num_classes = boxes_3d.shape[1]
    points_xyz = np.expand_dims(points_xyz, axis=1)
    points_xyz = np.tile(points_xyz, (1, num_classes, 1))
    encoded_boxes_3d[:, :, 0] = boxes_3d[:, :, 0] - points_xyz[:, :, 0]
    encoded_boxes_3d[:, :, 1] = boxes_3d[:, :, 1] - points_xyz[:, :, 1]
    encoded_boxes_3d[:, :, 2] = boxes_3d[:, :, 2] - points_xyz[:, :, 2]
    for cls_name in label_map:
        if cls_name == "Background" or cls_name == "DontCare":
            continue
        cls_label = label_map[cls_name]
        l, h, w = median_object_size_map[cls_name]

        mask = cls_labels[:, 0] == cls_label
        encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/l
        encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/h
        encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/w
        encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/l)
        encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/h)
        encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/w)
        encoded_boxes_3d[mask, 0, 6] = boxes_3d[mask, 0, 6]/(np.pi*0.25)
        # vertical
        mask = cls_labels[:, 0] == (cls_label+1)
        encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/l
        encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/h
        encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/w
        encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/l)
        encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/h)
        encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/w)
        encoded_boxes_3d[mask, 0, 6] = (boxes_3d[mask, 0, 6]-np.pi*0.5)/(np.pi*0.25)

    return encoded_boxes_3d

def box_decoding(cls_labels, points_xyz, encoded_boxes,
    label_map):
    decoded_boxes_3d = np.copy(encoded_boxes)
    for cls_name in label_map:
        if cls_name == "Background" or cls_name == "DontCare":
            continue
        cls_label = label_map[cls_name]
        l, h, w = median_object_size_map[cls_name]
        # Car horizontal
        mask = cls_labels[:, 0] == cls_label
        decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*l
        decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*h
        decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*w
        decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*l
        decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*h
        decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*w
        decoded_boxes_3d[mask, 0, 6] = encoded_boxes[mask, 0, 6]*(np.pi*0.25)
        # Car vertical
        mask = cls_labels[:, 0] == (cls_label+1)
        decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*l
        decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*h
        decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*w
        decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*l
        decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*h
        decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*w
        decoded_boxes_3d[mask, 0, 6] = (encoded_boxes[mask, 0, 6])*(np.pi*0.25)+0.5*np.pi

    # offset
    num_classes = encoded_boxes.shape[1]
    points_xyz = np.expand_dims(points_xyz, axis=1)
    points_xyz = np.tile(points_xyz, (1, num_classes, 1))
    decoded_boxes_3d[:, :, 0] = decoded_boxes_3d[:, :, 0] + points_xyz[:, :, 0]
    decoded_boxes_3d[:, :, 1] = decoded_boxes_3d[:, :, 1] + points_xyz[:, :, 1]
    decoded_boxes_3d[:, :, 2] = decoded_boxes_3d[:, :, 2] + points_xyz[:, :, 2]

    return decoded_boxes_3d

